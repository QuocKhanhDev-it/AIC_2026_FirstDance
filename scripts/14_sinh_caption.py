"""
14_sinh_caption.py — Sinh caption tiếng Việt cho keyframe bằng VLM (KÊNH 5).

Kênh 5 lấp lỗ hổng A8.4: metadata mô tả *cả video*, objects cho *nhãn rời rạc*,
không kênh nào diễn tả được **quan hệ trong một cảnh**. Truy vấn kiểu *"công
trình dạng vòng elip bằng gạch đất nung"* trượt ở cả bốn kênh còn lại.

Script này chỉ sinh văn bản. Phần truy hồi là `bm25.KenhVanBan.tu_bang_khung`
— cùng bộ máy với kênh 2 và 3, không viết lại.

⚠️ CAPTION PHẢI LÀ TIẾNG VIỆT — ĐÂY KHÔNG PHẢI LỰA CHỌN THẨM MỸ
================================================================

BM25 khớp **mặt chữ**, không hiểu nghĩa. Caption tiếng Anh + truy vấn tiếng
Việt = khớp đúng 0 token, kênh im lặng trả về rỗng. Không có không gian vector
chung nào để bắc cầu như bên CLIP.

Và ta đã xem đúng bộ phim này rồi: kênh 1 được **0,0000** trên tập dev vì CLIP
mù tiếng Việt (A10). Sinh caption tiếng Anh là tự dựng lại cùng một lỗi, lần
này tốn thêm tiền API.

CHI PHÍ — ƯỚC TRƯỚC KHI TIÊU
=============================

`--uoc-tinh` in ra số lệnh gọi, thời gian tường và chi phí, **không gọi API lần
nào**. Đơn giá là THAM SỐ (`--gia-1k`) chứ không phải hằng số chôn trong code:
giá nhà cung cấp đổi, và một con số bịa trong tài liệu còn tệ hơn không có số.
Tra bảng giá hiện hành rồi truyền vào.

    python scripts/14_sinh_caption.py --uoc-tinh --chon co-anh
    python scripts/14_sinh_caption.py --chon nhom:L21 --n 200      # thử 200 ảnh
    python scripts/14_sinh_caption.py --chon co-anh --luong 8

Ghi nối vào `index/caption.jsonl` (an toàn khi ngắt giữa chừng), rồi biên ra
`index/caption.parquet`. Chạy lại là bỏ qua ảnh đã xong, không gọi lại.
"""

import argparse
import base64
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

API = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "{model}:generateContent?key={key}")

MODEL_MAC_DINH = "gemini-3.1-flash-lite"     # đã chốt ở D0.3

# Nhắc: viết cho MÁY TÌM KIẾM đọc, không viết cho người đọc. Nghĩa là ưu tiên
# danh từ cụ thể và động từ, bỏ hết chữ đưa đẩy ("bức ảnh này cho thấy...") vì
# BM25 đếm token — chữ thừa làm loãng tài liệu và hạ điểm MỌI token thật.
NHAC = """Mô tả cảnh trong ảnh bằng tiếng Việt, để phục vụ tìm kiếm video.

Nêu cụ thể, theo thứ tự: người (số lượng, giới tính, trang phục), hành động
đang diễn ra, đồ vật nổi bật, bối cảnh (trong nhà/ngoài trời, loại địa điểm),
và chữ xuất hiện trên hình nếu đọc được.

Yêu cầu:
- 2 đến 3 câu, tiếng Việt có dấu.
- Dùng danh từ và động từ cụ thể. KHÔNG viết "bức ảnh cho thấy", "trong hình có".
- Không suy đoán điều không nhìn thấy. Không bình luận.
- Trả lời thẳng nội dung mô tả, không thêm gì khác."""

# Nới từ vựng (document expansion). Xem giải thích ở `--dong-nghia`.
THEM_DONG_NGHIA = """
- Với những vật có nhiều cách gọi vùng miền, thêm các cách gọi kia trong ngoặc
  ngay sau lần nhắc đầu: "quả dứa (thơm, khóm)", "con lợn (heo)", "cái bát
  (chén)". Chỉ làm với vật chính, tối đa 3 lần trong cả đoạn."""


def nhac(dong_nghia: bool = True) -> str:
    """Câu nhắc gửi cho VLM.

    VÌ SAO CÓ NÚT `dong_nghia`. BM25 khớp **mặt chữ**: truy vấn nói "dứa" mà
    caption viết "khóm" thì điểm bằng 0, không có gì bắc cầu. Kho này có đủ cả
    ba cách gọi — 220 video dùng "dứa", 76 dùng "thơm", 3 dùng "khóm" (tiêu đề
    L27 đúng là *"Trăm Năm làng Khóm"*).

    Metadata thì phải chịu vì nó cho sẵn. **Caption thì ta viết ra**, nên sửa
    được ngay ở đây, không tốn thêm lượt gọi nào. Đây là kỹ thuật *document
    expansion* cổ điển của IR.

    Nhưng nó KHÔNG miễn phí, và phải nói rõ: thêm chữ là kéo dài tài liệu, mà
    BM25 **phạt tài liệu dài** — đúng lý do hàm `don()` tồn tại. Nới quá tay thì
    hạ điểm mọi token thật. Vì vậy giới hạn "vật chính, tối đa 3 lần".

    Bật mặc định vì document expansion là kỹ thuật đã được chứng minh rộng rãi,
    NHƯNG chưa đo được trên kho này (chưa có khóa API). Lần chạy thật đầu tiên
    hãy A/B trên 4.086 ảnh thử: sinh hai lượt, `--dong-nghia` và `--khong-dong-
    nghia`, rồi chấm bằng `cham_diem.bao_cao_do_nhay`.
    """
    return NHAC + (THEM_DONG_NGHIA if dong_nghia else "")


def chon_row(master: pd.DataFrame, chon: str) -> pd.DataFrame:
    """Chọn tập keyframe cần sinh caption.

    Chỉ lấy dòng CÓ ẢNH trên máy này. Không có ảnh thì không có gì để nhìn —
    và mỗi máy giữ một phần kho khác nhau (B4), nên tập chọn được là khác nhau
    tùy máy. Đó là lý do có `--chon`, thay vì mặc định làm cả kho.
    """
    m = master[master.kf_path.notna()]
    if chon == "co-anh":
        return m
    if chon.startswith("nhom:"):
        nhom = {x.strip().upper() for x in chon.split(":", 1)[1].split(",")}
        return m[m.video_id.str[:3].isin(nhom)]
    if chon == "tap-dev":
        import tap_dev
        cau = tap_dev.doc() + tap_dev.doc(tap_dev.MAC_DINH_TEST)
        vid = set()
        for c in cau:
            p = ([r for b in c.row_id_dung for r in b]
                 if c.loai == "TRAKE" else c.row_id_dung)
            vid |= {master.video_id.iloc[r] for r in p}
        return m[m.video_id.isin(vid)]
    raise SystemExit(f"--chon không hiểu: {chon!r}. "
                     f"Dùng: co-anh | tap-dev | nhom:L21,L22")


def doc_log(f: Path) -> list[dict]:
    """Đọc `caption.jsonl`, **bỏ qua dòng hỏng**.

    ⚠️ Phải tự đọc chứ không dùng `pd.read_json(lines=True)`: ngắt giữa lúc ghi
    (Ctrl+C, mất mạng, hết pin) để lại một dòng cụt, và `read_json` ném
    `ValueError` cho cả file. Đã vấp: `da_xong()` chịu được dòng cụt nên lần
    chạy tiếp vẫn nối được, nhưng `bien()` lại chết ở cuối — tức đúng lúc đã
    tiêu xong tiền API thì không lấy ra được `caption.parquet`.
    """
    if not f.exists():
        return []
    ra, hong = [], 0
    for d in f.read_text("utf-8").splitlines():
        if not d.strip():
            continue
        try:
            x = json.loads(d)
            ra.append({"row_id": int(x["row_id"]), "caption": x.get("caption", "")})
        except Exception:
            hong += 1
    if hong:
        print(f"⚠️  bỏ qua {hong} dòng hỏng trong {f.name} (ngắt giữa lúc ghi)")
    return ra


def da_xong(f: Path) -> set:
    """`row_id` đã có caption."""
    return {x["row_id"] for x in doc_log(f)}


def goi(model, key, anh: bytes, loi_nhac: str, timeout=90, thu_lai=4):
    """Gọi API, lùi thời gian theo cấp số nhân khi 429/503."""
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/jpeg",
                         "data": base64.b64encode(anh).decode()}},
        {"text": loi_nhac}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}}
    data = json.dumps(body).encode()
    cho = 5
    for lan in range(thu_lai):
        req = urllib.request.Request(API.format(model=model, key=key), data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            try:
                return d["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                return ""                    # bị chặn / rỗng
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and lan < thu_lai - 1:
                time.sleep(cho)
                cho *= 2
                continue
            raise
    raise RuntimeError("hết lượt thử lại")


def don(text: str) -> str:
    """Bỏ phần mở đầu thừa mà model hay thêm dù đã bảo đừng.

    Không bỏ thì mọi caption đều chứa "bức ảnh cho thấy" — token đó có mặt ở
    100% tài liệu nên IDF ~ 0, vô hại về điểm, nhưng nó kéo dài tài liệu và
    BM25 PHẠT tài liệu dài. Tức là chữ thừa hạ điểm mọi token thật.
    """
    t = " ".join(text.split())
    for mo in ("bức ảnh cho thấy", "hình ảnh cho thấy", "trong ảnh có",
               "trong hình có", "bức ảnh này", "hình này", "ảnh cho thấy"):
        if t.lower().startswith(mo):
            t = t[len(mo):].lstrip(" ,:.")
            break
    return t


def uoc_tinh(d: pd.DataFrame, a):
    """In bảng ước lượng. KHÔNG gọi API."""
    n = len(d)
    giay = n * a.giay_moi_anh / max(a.luong, 1)
    print(f"{'số ảnh cần sinh':<28}{n:>12,}")
    print(f"{'luồng song song':<28}{a.luong:>12}")
    print(f"{'giây/ảnh (giả định)':<28}{a.giay_moi_anh:>12.1f}")
    print(f"{'thời gian tường':<28}{giay / 3600:>12.1f} giờ")
    print(f"{'chi phí ở ' + str(a.gia_1k) + ' /1k ảnh':<28}"
          f"{n / 1000 * a.gia_1k:>12.2f}")
    print(f"\n{'toàn kho 177.321 ảnh':<28}"
          f"{177321 * a.giay_moi_anh / max(a.luong, 1) / 3600:>12.1f} giờ"
          f"   {177321 / 1000 * a.gia_1k:>10.2f}")
    print("\n⚠️  `--gia-1k` là THAM SỐ, mặc định chỉ là chỗ giữ chỗ. Tra bảng giá\n"
          "    hiện hành rồi truyền vào — đừng trích con số này ra tài liệu.")
    print("⚠️  Bậc miễn phí có trần lượt/ngày. Ba model đã chết vì rate-limit\n"
          "    trong một đợt test 20 lượt (D0.3). Với vài chục nghìn ảnh thì\n"
          "    phải trả phí hoặc chạy model local — đó là việc 12 của PHẦN H.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--chon", default="tap-dev",
                    help="co-anh | tap-dev | nhom:L21,L22")
    ap.add_argument("--model", default=MODEL_MAC_DINH)
    ap.add_argument("--n", type=int, default=0, help="trần số ảnh lần này. 0 = hết")
    ap.add_argument("--luong", type=int, default=4, help="số luồng gọi song song")
    ap.add_argument("--uoc-tinh", action="store_true", help="chỉ ước lượng, không gọi API")
    ap.add_argument("--giay-moi-anh", type=float, default=1.4,
                    help="độ trễ đo được ở D0.3")
    ap.add_argument("--gia-1k", type=float, default=0.0,
                    help="đơn giá cho 1000 ảnh, tự tra rồi truyền vào")
    ap.add_argument("--bien", action="store_true",
                    help="chỉ biên caption.jsonl -> caption.parquet rồi thoát")
    ap.add_argument("--khong-dong-nghia", action="store_true",
                    help="tắt nới từ vựng vùng miền (dứa/thơm/khóm). "
                         "Để A/B trên tập thử — xem hàm nhac()")
    a = ap.parse_args()
    loi_nhac = nhac(not a.khong_dong_nghia)

    log = a.index / "caption.jsonl"
    par = a.index / "caption.parquet"

    if a.bien:
        return bien(log, par)

    master = pd.read_parquet(a.index / "master.parquet")
    d = chon_row(master, a.chon)
    xong = da_xong(log)
    d = d[~d.row_id.isin(xong)]
    if a.n:
        d = d.head(a.n)

    print(f"chọn '{a.chon}': còn {len(d):,} ảnh cần sinh"
          f"  (đã xong {len(xong):,})\n")
    if a.uoc_tinh:
        return uoc_tinh(d, a)
    if d.empty:
        print("Không còn ảnh nào. Biên ra parquet:")
        return bien(log, par)

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "Chưa đặt GOOGLE_API_KEY.\n\n"
            "    $env:GOOGLE_API_KEY = '...'\n\n"
            "Đặt bằng biến môi trường, ĐỪNG ghi vào file trong repo — repo này\n"
            "công khai. Và khóa nào từng dán vào chat thì coi như đã lộ, phải\n"
            "xoá ở AI Studio rồi tạo khóa mới.")

    viec = queue.Queue()
    for r in d.itertuples(index=False):
        viec.put((int(r.row_id), r.kf_path))
    khoa = threading.Lock()
    dem = {"xong": 0, "hong": 0}
    t0 = time.perf_counter()

    def chay():
        while True:
            try:
                rid, path = viec.get_nowait()
            except queue.Empty:
                return
            try:
                cap = don(goi(a.model, key, Path(path).read_bytes(), loi_nhac))
            except Exception as e:
                with khoa:
                    dem["hong"] += 1
                    print(f"  hỏng row_id {rid}: {str(e)[:70]}", flush=True)
                continue
            with khoa:
                # Ghi NỐI từng dòng: ngắt giữa chừng vẫn giữ nguyên phần đã làm.
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"row_id": rid, "caption": cap},
                                       ensure_ascii=False) + "\n")
                dem["xong"] += 1
                if dem["xong"] % 25 == 0:
                    t = time.perf_counter() - t0
                    con = (len(d) - dem["xong"]) * t / dem["xong"] / 60
                    print(f"  {dem['xong']:,}/{len(d):,}  "
                          f"{dem['xong'] / t:.1f} ảnh/giây  còn ~{con:.0f} phút",
                          flush=True)

    luong = [threading.Thread(target=chay, daemon=True) for _ in range(a.luong)]
    for t in luong:
        t.start()
    for t in luong:
        t.join()

    giay = time.perf_counter() - t0
    print(f"\nXong {dem['xong']:,} ảnh trong {giay / 60:.1f} phút"
          f"  ({dem['xong'] / giay:.1f} ảnh/giây)"
          + (f"  |  {dem['hong']} hỏng" if dem["hong"] else ""))
    bien(log, par)


def bien(log: Path, par: Path):
    """`caption.jsonl` -> `caption.parquet`, bỏ trùng, giữ bản cuối."""
    if not log.exists():
        raise SystemExit(f"Chưa có {log}")
    dong = doc_log(log)
    if not dong:
        raise SystemExit(f"{log} không có dòng nào đọc được")
    d = pd.DataFrame(dong)
    d = d.drop_duplicates("row_id", keep="last").sort_values("row_id")
    d = d[d.caption.fillna("").str.strip() != ""]
    d.to_parquet(par, index=False)
    print(f"\n{par}: {len(d):,} caption"
          f"  |  trung bình {int(d.caption.str.len().mean())} ký tự")
    print("\nDùng ngay:")
    print("    from bm25 import KenhVanBan")
    print("    k5 = KenhVanBan.tu_bang_khung(master, "
          "pd.read_parquet('index/caption.parquet'))")
    return 0


if __name__ == "__main__":
    main()
