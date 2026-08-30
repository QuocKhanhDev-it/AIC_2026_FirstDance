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

HAI BACKEND — CHỐT DÙNG FREE, KHÔNG TRẢ PHÍ
=============================================

Nhóm đã chốt chỉ dùng model free. Ở khối lượng kênh 5 (tới 177.321 ảnh), free
tier Gemini có trần lượt/phút + lượt/ngày — chạy đúng trần đó cho hết một phần
kho có thể mất hàng tuần. `qwen2.5vl:7b` qua Ollama (local, GPU) không có trần
nào, chỉ bị giới hạn bởi tốc độ GPU (~8s/ảnh đo được ở bench Q&A, xem
`Test VLM AIC/BAO_CAO_BENCH_VLM_2026-08-13.md`) — CHẬM HƠN nhưng CHẠY ĐƯỢC HẾT.

    --backend ollama     # NGUỒN CHÍNH cho việc sinh hàng loạt (mặc định)
    --backend gemini      # bổ sung cho vài trăm ảnh khó, hoặc khi GPU đang bận

⚠️ Ollama chưa từng được đo cho việc VIẾT CAPTION DÀI (bench cũ chỉ đo trả lời
ngắn ≤4 từ cho Q&A) — bắt buộc thử trên mẫu nhỏ trước, xem CHI PHÍ bên dưới.

CHI PHÍ / THỜI GIAN — ƯỚC TRƯỚC KHI CHẠY THẬT
================================================

`--uoc-tinh` in ra số lệnh gọi và thời gian tường, **không gọi API/Ollama lần
nào**. Với `--backend gemini`, `--gia-1k` là THAM SỐ (đơn giá $ / 1000 ảnh) chứ
không phải hằng số chôn trong code — giá nhà cung cấp đổi, một con số bịa trong
tài liệu còn tệ hơn không có số.

    python scripts/14_sinh_caption.py --uoc-tinh --chon co-anh
    python scripts/14_sinh_caption.py --chon nhom:L21 --n 200                 # thử 200 ảnh, Ollama
    python scripts/14_sinh_caption.py --chon co-anh --backend gemini --luong 8

⚠️ Ollama chạy trên MỘT GPU — `--luong` cao không chia sẻ được VRAM, dễ tràn bộ
nhớ hoặc không nhanh hơn. Bắt đầu với `--luong 1` hoặc `2`, tự đo trước khi tăng.

Ghi nối vào `index/caption.jsonl` (an toàn khi ngắt giữa chừng), rồi biên ra
`index/caption.parquet`. Chạy lại là bỏ qua ảnh đã xong, không gọi lại.
"""

import argparse
import base64
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

API_GEMINI = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent?key={key}")
API_OLLAMA = "http://localhost:11434/api/generate"

# Chốt dùng FREE (không trả phí): Ollama làm nguồn chính cho khối lượng lớn,
# Gemini free tier bổ sung. Xem "HAI BACKEND" ở đầu file.
MODEL_MAC_DINH = {"gemini": "gemini-3.1-flash-lite", "ollama": "qwen2.5vl:7b",
                  "hf": "Qwen/Qwen2.5-VL-3B-Instruct"}
GIAY_MOI_ANH_MAC_DINH = {"gemini": 1.4, "ollama": 8.2,   # đo được ở bench Q&A
                         "hf": 1.5}                      # ƯỚC, chưa đo — xem chay_hf

# Nhắc: viết cho MÁY TÌM KIẾM đọc, không viết cho người đọc. Nghĩa là ưu tiên
# danh từ cụ thể và động từ, bỏ hết chữ đưa đẩy ("bức ảnh này cho thấy...") vì
# BM25 đếm token — chữ thừa làm loãng tài liệu và hạ điểm MỌI token thật.
NHAC = """Mô tả cảnh trong ảnh bằng tiếng Việt, để phục vụ tìm kiếm video.

Nêu cụ thể, theo thứ tự: người (số lượng, giới tính, trang phục), hành động
đang diễn ra, đồ vật nổi bật, bối cảnh (trong nhà/ngoài trời, loại địa điểm),
và chữ xuất hiện trên hình nếu đọc được.

Yêu cầu:
- CHỈ dùng tiếng Việt. Tuyệt đối không chèn từ, cụm từ hay câu bằng tiếng Anh,
  tiếng Tây Ban Nha hay bất kỳ ngôn ngữ nào khác.
- 2 đến 3 câu, tiếng Việt có dấu. Không lặp lại cùng một câu hay cùng một ý.
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
    if chon.startswith("tap:"):
        # Cùng phép chọn với `08_encode.py --theo-tap-dev`: TRỌN VẸN mọi video
        # mà tập đó đụng tới, không phải vài ảnh quanh đáp án. Bể ứng viên hẹp
        # lại quanh đáp án là tự thổi phồng điểm.
        rid = []
        for dong in Path(chon.split(":", 1)[1]).read_text("utf-8").splitlines():
            if not dong.strip():
                continue
            r = json.loads(dong)["row_id_dung"]
            rid += [x for b in r for x in b] if isinstance(r[0], list) else r
        return m[m.video_id.isin(set(master.video_id.iloc[rid]))]
    raise SystemExit(f"--chon không hiểu: {chon!r}. "
                     f"Dùng: co-anh | tap-dev | tap:<file.jsonl> | nhom:L21,L22")


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


def goi_gemini(model, key, anh: bytes, loi_nhac: str, timeout=90, thu_lai=4):
    """Gọi Gemini API, lùi thời gian theo cấp số nhân khi 429/503."""
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/jpeg",
                         "data": base64.b64encode(anh).decode()}},
        {"text": loi_nhac}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}}
    data = json.dumps(body).encode()
    cho = 5
    for lan in range(thu_lai):
        req = urllib.request.Request(API_GEMINI.format(model=model, key=key), data=data,
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


def goi_ollama(model, anh: bytes, loi_nhac: str, timeout=120, thu_lai=3, nhiet_do=0.0):
    """Gọi Ollama local (`ollama serve`), lùi thời gian khi server bận/chưa lên.

    Không có khái niệm 429 ở đây — lỗi thường gặp là server chưa chạy hoặc
    model chưa `ollama pull`, nên thông báo lỗi phải nói rõ cách sửa, không chỉ
    ném traceback.
    """
    # temperature=0.0 không có trần token thì thỉnh thoảng rơi vào lặp vô hạn
    # (đo được: 1 caption 9.872 ký tự, lặp cùng một câu hàng trăm lần — BM25
    # phạt tài liệu dài nên lỗi này tự hại kênh 5). num_predict chặn trần,
    # repeat_penalty giảm khả năng rơi vào vòng lặp từ đầu.
    body = {"model": model, "prompt": loi_nhac, "stream": False,
             "images": [base64.b64encode(anh).decode()],
             "options": {"temperature": nhiet_do, "num_predict": 220,
                        "repeat_penalty": 1.3}}
    data = json.dumps(body).encode()
    cho = 3
    for lan in range(thu_lai):
        req = urllib.request.Request(API_OLLAMA, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            return d.get("response", "").strip()
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError) as e:
            if lan < thu_lai - 1:
                time.sleep(cho)
                cho *= 2
                continue
            raise RuntimeError(
                f"Ollama không phản hồi ở {API_OLLAMA} sau {thu_lai} lần thử "
                f"({e}). Đã chạy `ollama serve` và `ollama pull {model}` chưa?")
    raise RuntimeError("hết lượt thử lại")


CHU_HAN = re.compile(r"[一-鿿぀-ヿ가-힯]")


def goi_ollama_sach(model, anh: bytes, loi_nhac: str, timeout=120, thu_lai=3):
    """`goi_ollama`, tự gọi lại nếu kết quả lẫn chữ Hán/Nhật/Hàn.

    Đo được trên 200 ảnh (19/08): qwen2.5vl:7b lẫn chữ Hán vào ~12% caption —
    KHÔNG ngẫu nhiên, tập trung ở một loại cảnh (bàn dẫn chương trình tin tức)
    lặp lại y hệt qua nhiều khung liên tiếp. Sửa câu nhắc (thêm "chỉ tiếng
    Việt") KHÔNG giảm được tỷ lệ này — vì `temperature=0.0` là argmax, cùng
    ảnh cùng prompt luôn đi lại đúng một đường sinh. Phải đổi temperature mới
    có cơ hội thoát đường đó, không phải đổi chữ trong prompt.
    """
    cap = goi_ollama(model, anh, loi_nhac, timeout=timeout, thu_lai=thu_lai, nhiet_do=0.0)
    for nhiet in (0.5, 0.9):
        if not CHU_HAN.search(cap):
            return cap
        cap = goi_ollama(model, anh, loi_nhac, timeout=timeout, thu_lai=thu_lai, nhiet_do=nhiet)
    if CHU_HAN.search(cap):
        # Vài cảnh (bàn dẫn chương trình) kéo model về tiếng Trung rất mạnh,
        # 2 lần thử lại không thoát được. Lưới an toàn cuối: xóa ký tự Hán
        # còn sót — mất một phần nội dung ở đúng chỗ đó, nhưng còn hơn để
        # ký tự vô nghĩa với BM25 tiếng Việt lọt vào caption.
        cap = " ".join(CHU_HAN.sub("", cap).split())
    return cap


def goi(backend, model, key, anh: bytes, loi_nhac: str, timeout=90, thu_lai=4):
    """Điều phối theo backend. `key` bị bỏ qua khi `backend == "ollama"`."""
    if backend == "ollama":
        return goi_ollama_sach(model, anh, loi_nhac, timeout=max(timeout, 120), thu_lai=thu_lai)
    return goi_gemini(model, key, anh, loi_nhac, timeout=timeout, thu_lai=thu_lai)


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


def chay_hf(d: pd.DataFrame, log: Path, loi_nhac: str, ten_model: str,
            batch: int, diem_anh: int, so_chu: int):
    """Sinh caption bằng model nạp THẲNG trong tiến trình (dùng cho Kaggle GPU).

    VÌ SAO KHÔNG DÙNG CHUNG ĐƯỜNG `goi()` NHƯ HAI BACKEND KIA. Ollama và Gemini
    là *server*: bắn nhiều luồng vào là chúng tự xếp hàng và tự gộp lô. Model
    nạp tại chỗ thì không — `--luong` cao chỉ khiến nhiều thread giành một GPU,
    chậm hơn chứ không nhanh hơn. Cách duy nhất tăng thông lượng là **gộp lô
    thật** (`--batch`), nên đường chạy này tuần tự và không đụng hàng đợi.

    HAI NÚT QUYẾT ĐỊNH TỐC ĐỘ, cả hai đều KHÔNG phải `--batch`:

    * `--diem-anh` — trần số điểm ảnh đưa vào bộ mã hoá thị giác. Qwen2.5-VL
      chia ảnh thành token theo diện tích, để mặc định thì một ảnh 1280×720 tốn
      hàng nghìn token thị giác. Đây là nút đắt nhất.
    * `--so-chu` — trần token sinh ra. Caption 2–3 câu không cần quá ~180.

    ⚠️ **Chưa có số đo trên phần cứng Kaggle.** `GIAY_MOI_ANH_MAC_DINH["hf"]`
    là con số ƯỚC, không phải đo. Chạy `--n 40` trước và đọc tốc độ thật in ra,
    rồi mới nhân lên để quyết định phạm vi — đừng tin dòng `--uoc-tinh` khi
    backend là `hf`.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"nạp {ten_model} …", flush=True)
    proc = AutoProcessor.from_pretrained(
        ten_model, min_pixels=256 * 28 * 28, max_pixels=diem_anh * 28 * 28)
    # padding TRÁI: `generate` lấy token cuối của mỗi dòng làm mốc sinh tiếp.
    # Pad phải là sinh tiếp từ ô đệm -> cả lô ra rác, và KHÔNG có lỗi nào báo.
    proc.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ten_model, torch_dtype=torch.float16, device_map="auto").eval()

    from qwen_vl_utils import process_vision_info
    hang = list(d.itertuples(index=False))
    t0 = time.perf_counter()
    xong = hong = 0

    for i in range(0, len(hang), batch):
        lo = hang[i:i + batch]
        tin = [[{"role": "user", "content": [
                    {"type": "image", "image": str(r.kf_path)},
                    {"type": "text", "text": loi_nhac}]}] for r in lo]
        try:
            van = [proc.apply_chat_template(t, tokenize=False,
                                            add_generation_prompt=True) for t in tin]
            anh, vid = process_vision_info(tin)
            vao = proc(text=van, images=anh, videos=vid,
                       padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                ra = model.generate(**vao, max_new_tokens=so_chu, do_sample=False)
            tra = proc.batch_decode([o[len(v):] for v, o in zip(vao.input_ids, ra)],
                                    skip_special_tokens=True)
        except Exception as e:
            # Cả lô hỏng (thường là tràn VRAM). Báo rõ rồi đi tiếp — mất một lô
            # còn hơn chết cả lượt chạy đã làm được vài nghìn ảnh.
            hong += len(lo)
            print(f"  hỏng lô {i}-{i + len(lo)}: {str(e)[:90]}", flush=True)
            continue

        with log.open("a", encoding="utf-8") as f:
            for r, t in zip(lo, tra):
                # Không thử lại đổi nhiệt độ như `goi_ollama_sach`: ở đây greedy
                # là cả lô, sinh lại một ảnh là phá nhịp gộp lô. Xoá thẳng ký tự
                # Hán — thà mất một phần nội dung còn hơn để token vô nghĩa với
                # BM25 tiếng Việt lọt vào caption.
                cap = don(" ".join(CHU_HAN.sub("", t).split()))
                f.write(json.dumps({"row_id": int(r.row_id), "caption": cap},
                                   ensure_ascii=False) + "\n")
        xong += len(lo)
        giay = time.perf_counter() - t0
        con = (len(hang) - xong) * giay / xong / 60
        print(f"  {xong:,}/{len(hang):,}  {giay / xong:.2f} giây/ảnh  "
              f"còn ~{con:.0f} phút", flush=True)

    print(f"\nXong {xong:,} ảnh trong {(time.perf_counter() - t0) / 60:.1f} phút"
          + (f"  |  {hong} hỏng" if hong else ""))


def uoc_tinh(d: pd.DataFrame, a):
    """In bảng ước lượng. KHÔNG gọi API/Ollama."""
    n = len(d)
    # `hf` nạp model TẠI CHỖ: một GPU, chạy tuần tự theo lô. Chia cho `--luong`
    # ở đây là bịa ra tốc độ gấp 4 lần thực tế.
    chia = 1 if a.backend == "hf" else max(a.luong, 1)
    giay = n * a.giay_moi_anh / chia
    print(f"{'backend':<28}{a.backend:>12}")
    print(f"{'model':<28}{a.model:>12}")
    print(f"{'số ảnh cần sinh':<28}{n:>12,}")
    print(f"{'song song':<28}"
          f"{('lô ' + str(a.batch)) if a.backend == 'hf' else a.luong:>12}")
    print(f"{'giây/ảnh (giả định)':<28}{a.giay_moi_anh:>12.1f}")
    print(f"{'thời gian tường':<28}{giay / 3600:>12.1f} giờ")
    print(f"\n{'toàn kho 177.321 ảnh':<28}"
          f"{177321 * a.giay_moi_anh / chia / 3600:>12.1f} giờ")
    if a.backend == "gemini":
        print(f"{'chi phí ở ' + str(a.gia_1k) + ' /1k ảnh':<28}"
              f"{n / 1000 * a.gia_1k:>12.2f}")
        print(f"{'chi phí toàn kho':<28}{177321 / 1000 * a.gia_1k:>12.2f}")
        print("\n⚠️  `--gia-1k` là THAM SỐ, mặc định chỉ là chỗ giữ chỗ. Tra bảng giá\n"
              "    hiện hành rồi truyền vào — đừng trích con số này ra tài liệu.")
        print("⚠️  Free tier có trần lượt/phút + lượt/ngày, chưa đo trần thật cho\n"
              "    model này. Ở khối lượng vài chục nghìn ảnh, dùng `--backend ollama`\n"
              "    (mặc định) để không bị nghẽn quota — Gemini chỉ nên bổ sung.")
    elif a.backend == "hf":
        print("\n⚠️  1,5 giây/ảnh là con số ƯỚC, CHƯA ĐO trên phần cứng Kaggle —\n"
              "    bảng trên chỉ là chỗ giữ chỗ. Chạy `--n 40` trước, đọc tốc độ\n"
              "    THẬT script in ra, rồi mới nhân lên để quyết định phạm vi.")
        print("⚠️  Quota Kaggle 30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ. Con số\n"
              "    'toàn kho' ở trên vượt cả hai — phải chọn tập con.")
    else:
        print("\n⚠️  Ollama chạy trên MỘT GPU — `--luong` cao không chia sẻ được VRAM,\n"
              "    dễ tràn bộ nhớ. Bắt đầu `--luong 1` hoặc `2`, tự đo trước khi tăng.")
        print("⚠️  Chưa từng đo qwen2.5vl:7b cho việc VIẾT CAPTION DÀI (bench cũ chỉ\n"
              "    đo trả lời ngắn cho Q&A) — chạy `--n 200` trước, chấm bằng\n"
              "    `cham_diem.bao_cao_do_nhay` trước khi cam kết chạy hết.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--chon", default="tap-dev",
                    help="co-anh | tap-dev | tap:<file.jsonl> | nhom:L21,L22")
    ap.add_argument("--backend", default="ollama",
                    choices=("ollama", "gemini", "hf"),
                    help="ollama = nguồn chính (free, local, không trần). "
                         "gemini = bổ sung (free tier, có trần). "
                         "hf = model nạp tại chỗ, cho GPU Kaggle")
    ap.add_argument("--batch", type=int, default=8,
                    help="chỉ dùng với --backend hf: số ảnh mỗi lô")
    ap.add_argument("--diem-anh", type=int, default=512,
                    help="chỉ dùng với --backend hf: trần token thị giác mỗi "
                         "ảnh. Nút đắt nhất về tốc độ — xem chay_hf()")
    ap.add_argument("--so-chu", type=int, default=180,
                    help="chỉ dùng với --backend hf: trần token sinh ra")
    ap.add_argument("--model", default=None,
                    help="mặc định theo backend, xem MODEL_MAC_DINH")
    ap.add_argument("--n", type=int, default=0, help="trần số ảnh lần này. 0 = hết")
    ap.add_argument("--luong", type=int, default=4,
                    help="số luồng song song. Ollama trên 1 GPU: bắt đầu 1-2")
    ap.add_argument("--uoc-tinh", action="store_true",
                    help="chỉ ước lượng, không gọi API/Ollama")
    ap.add_argument("--giay-moi-anh", type=float, default=None,
                    help="mặc định theo backend, xem GIAY_MOI_ANH_MAC_DINH")
    ap.add_argument("--gia-1k", type=float, default=0.0,
                    help="đơn giá cho 1000 ảnh (chỉ dùng khi --backend gemini)")
    ap.add_argument("--bien", action="store_true",
                    help="chỉ biên caption.jsonl -> caption.parquet rồi thoát")
    ap.add_argument("--khong-dong-nghia", action="store_true",
                    help="tắt nới từ vựng vùng miền (dứa/thơm/khóm). "
                         "Để A/B trên tập thử — xem hàm nhac()")
    a = ap.parse_args()
    if a.model is None:
        a.model = MODEL_MAC_DINH[a.backend]
    if a.giay_moi_anh is None:
        a.giay_moi_anh = GIAY_MOI_ANH_MAC_DINH[a.backend]
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

    if a.backend == "hf":
        chay_hf(d, log, loi_nhac, a.model, a.batch, a.diem_anh, a.so_chu)
        return bien(log, par)

    key = None
    if a.backend == "gemini":
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
                cap = don(goi(a.backend, a.model, key, Path(path).read_bytes(), loi_nhac))
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


def soat_ro_dap_an(rid: set, index: Path) -> None:
    """Cảnh báo khi tập ảnh đã caption dồn vào ĐÚNG khung đáp án tập dev.

    ⚠️ ĐÂY LÀ LOẠI HỎNG KHÔNG LÀM SAI DỮ LIỆU, CHỈ LÀM VÔ NGHĨA PHÉP ĐO.

    BM25 chỉ xếp hạng những tài liệu **tồn tại**. Ảnh chưa có caption thì không
    có tài liệu nào, nên kênh 5 không bao giờ trả về nó. Caption riêng khung
    đáp án là dựng một bể ứng viên mà đáp án chiếm phần lớn — kênh sẽ tìm ra
    đáp án vì nó gần như là thứ DUY NHẤT có mặt, không phải vì caption tả đúng.

    Đo được (30/08) trên `caption.jsonl` đầu tiên: 191/347 ảnh (**55%**) là
    khung đáp án, mật độ trong từng video chỉ 0,3–10%. Nếu chấm kênh 5 trên
    file đó thì điểm sẽ rất đẹp và không nói lên bất cứ điều gì.

    Cách đúng là quét **TRỌN VẸN** các video liên quan — chính là điều
    `--chon tap:<file>` và `--chon tap-dev` làm.
    """
    import glob as _glob
    dap_an = set()
    for f in _glob.glob(str(Path("dev") / "tap_*.jsonl")):
        for dong in Path(f).read_text("utf-8").splitlines():
            if not dong.strip():
                continue
            try:
                r = json.loads(dong)["row_id_dung"]
            except (KeyError, json.JSONDecodeError):
                continue
            dap_an |= set([x for b in r for x in b] if isinstance(r[0], list) else r)
    if not dap_an or not rid:
        return

    trung = len(rid & dap_an)
    ty_le = trung / len(rid)
    print(f"\nkhung đáp án tập dev trong phần đã caption: "
          f"{trung:,}/{len(rid):,} ({ty_le * 100:.1f}%)")
    if ty_le < 0.10:
        return
    print(
        f"\n⚠️  RÒ ĐÁP ÁN VÀO TẬP CAPTION — ĐỪNG CHẤM KÊNH 5 TRÊN FILE NÀY.\n"
        f"    {ty_le * 100:.0f}% ảnh đã caption là khung đáp án. BM25 chỉ xếp hạng\n"
        f"    tài liệu CÓ TỒN TẠI, nên kênh 5 sẽ tìm ra đáp án chỉ vì nó gần như\n"
        f"    là thứ duy nhất có caption — không phải vì caption tả đúng.\n\n"
        f"    Sửa: quét TRỌN VẸN các video liên quan, đừng chọn quanh đáp án.\n"
        f"      python scripts/14_sinh_caption.py --chon tap:dev/tap_de_that.jsonl\n"
        f"    Ảnh đã xong được bỏ qua, không sinh lại.\n")


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

    n_trung = len(dong) - len(d)
    if n_trung > 0:
        print(f"  ({n_trung} dòng trùng row_id đã bỏ, giữ bản cuối)")

    soat_ro_dap_an(set(d.row_id), par.parent)

    print("\nDùng ngay:")
    print("    from bm25 import KenhVanBan")
    print("    k5 = KenhVanBan.tu_bang_khung(master, "
          "pd.read_parquet('index/caption.parquet'))")
    return 0


if __name__ == "__main__":
    main()
