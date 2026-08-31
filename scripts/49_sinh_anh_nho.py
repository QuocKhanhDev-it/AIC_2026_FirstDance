"""
49_sinh_anh_nho.py — Sinh bản thu nhỏ của keyframe để máy yếu SOI ĐƯỢC ảnh.

    python scripts/49_sinh_anh_nho.py --uoc-tinh --chon nhom:L21
    python scripts/49_sinh_anh_nho.py --chon nhom:L26 --ghi          # trên Kaggle
    python scripts/49_sinh_anh_nho.py --chon co-anh --ghi --nen

VẤN ĐỀ NÓ GIẢI

L26 là **45% kho** và không máy nào trong nhóm có ảnh của nó — 12,13 GB, quá
lớn để ai cũng giữ một bản. Nhưng truy hồi **không cần ảnh**: ma trận vector đã
tính sẵn, `master.parquet` có đủ `frame_idx`. Ảnh chỉ cần cho đúng hai việc:

  1. Người soi bằng mắt xem ứng viên có đúng không.
  2. Trả lời câu Q&A.

Cả hai chỉ cần **nhận ra cảnh**, không cần độ nét gốc. Thu nhỏ còn ~256px là đủ
cho mắt người, mà nhẹ hơn cả chục lần.

VÌ SAO `--uoc-tinh` LÀ CHẾ ĐỘ MẶC ĐỊNH CỦA TÀI LIỆU NÀY

Dung lượng sau khi thu nhỏ **phụ thuộc nội dung ảnh** — cảnh nhiều chi tiết nén
kém hơn cảnh phẳng. Đoán một con số rồi ghi vào kế hoạch là cách chắc chắn để
sai. `--uoc-tinh` thu nhỏ thật một mẫu nhỏ, đo byte thật, rồi mới ngoại suy.

CẤU TRÚC RA — CỐ Ý GIỐNG HỆT CÂY KEYFRAME GỐC

    <ra>/L26_V001/001.jpg

Nhờ vậy `12_va_duong_dan.py` quét được nó y như thư mục ảnh thật (nó nhận diện
theo tên thư mục `L\\d\\d_V\\d+`), và `src/anh.py` chỉ cần ghép tên là ra đường
dẫn — không phải bảng tra nào cả.
"""

import argparse
import io
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))


def chon_row(master: pd.DataFrame, chon: str) -> pd.DataFrame:
    """Chỉ lấy dòng CÓ ẢNH GỐC trên máy này — không có gì để thu nhỏ thì thôi."""
    m = master[master.kf_path.notna()]
    if chon == "co-anh":
        return m
    if chon.startswith("nhom:"):
        nh = {x.strip().upper() for x in chon.split(":", 1)[1].split(",")}
        return m[m.video_id.str[:3].isin(nh)]
    if chon.startswith("video:"):
        vs = [v.strip() for v in
              Path(chon.split(":", 1)[1]).read_text("utf-8").splitlines() if v.strip()]
        return m[m.video_id.isin(vs)]
    raise SystemExit(f"--chon không hiểu: {chon!r}. "
                     f"Dùng: co-anh | nhom:L26 | video:<f.txt>")


def thu_nho(goc: Path, rong: int, chat: int) -> bytes:
    """Trả về JPEG đã thu nhỏ. Ảnh nhỏ hơn `rong` thì giữ nguyên kích cỡ."""
    from PIL import Image
    with Image.open(goc) as im:
        im = im.convert("RGB")
        if im.width > rong:
            cao = round(im.height * rong / im.width)
            im = im.resize((rong, cao), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=chat, optimize=True, progressive=True)
        return b.getvalue()


def uoc_tinh(d: pd.DataFrame, a) -> None:
    """Thu nhỏ THẬT một mẫu, đo byte thật, rồi ngoại suy."""
    import random
    mau = d.sample(min(a.so_mau, len(d)), random_state=0)
    t0 = time.perf_counter()
    goc_b = moi_b = 0
    hong = 0
    for r in mau.itertuples(index=False):
        p = Path(r.kf_path)
        try:
            goc_b += p.stat().st_size
            moi_b += len(thu_nho(p, a.rong, a.chat))
        except Exception:
            hong += 1
    giay = time.perf_counter() - t0
    n = len(mau) - hong
    if n == 0:
        raise SystemExit("❌ Không đọc được ảnh nào trong mẫu.")

    tb_goc, tb_moi = goc_b / n, moi_b / n
    print(f"{'mẫu đo được':<26}{n:>10,} ảnh"
          + (f"  ({hong} hỏng)" if hong else ""))
    print(f"{'rộng / chất lượng':<26}{a.rong:>10} px / {a.chat}")
    print(f"{'trung bình gốc':<26}{tb_goc / 1024:>10,.0f} KB")
    print(f"{'trung bình thu nhỏ':<26}{tb_moi / 1024:>10,.1f} KB"
          f"   (còn {tb_moi / tb_goc * 100:.1f}%)")
    print(f"{'tốc độ':<26}{n / giay:>10,.0f} ảnh/giây")
    print()
    print(f"{'phạm vi':<26}{'ảnh':>10}{'dung lượng':>14}{'thời gian':>12}")
    print("-" * 62)
    for ten, so in (("đang chọn", len(d)),
                    ("L26 (79.590)", 79_590),
                    ("toàn kho (177.321)", 177_321)):
        print(f"{ten:<26}{so:>10,}{so * tb_moi / 1024 ** 3:>12,.2f} GB"
              f"{so / (n / giay) / 60:>10,.0f} phút")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--chon", default="nhom:L26",
                    help="co-anh | nhom:L26 | video:<f.txt>")
    ap.add_argument("--ra", default=GOC / "index" / "anh_nho", type=Path)
    ap.add_argument("--rong", type=int, default=256,
                    help="chiều rộng đích, px. Ảnh hẹp hơn thì giữ nguyên")
    ap.add_argument("--chat", type=int, default=72, help="chất lượng JPEG")
    ap.add_argument("--n", type=int, default=0, help="trần số ảnh lần này. 0 = hết")
    ap.add_argument("--luong", type=int, default=8)
    ap.add_argument("--uoc-tinh", action="store_true",
                    help="thu nhỏ một mẫu, đo byte THẬT, không ghi gì")
    ap.add_argument("--so-mau", type=int, default=120)
    ap.add_argument("--ghi", action="store_true", help="ghi thật")
    ap.add_argument("--nen", action="store_true",
                    help="nén thư mục kết quả thành .zip sau khi xong")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    d = chon_row(master, a.chon)
    print(f"chọn '{a.chon}': {len(d):,} ảnh có sẵn trên máy này\n")
    if d.empty:
        raise SystemExit("Không có ảnh nào — máy này chưa tải nhóm đó?")

    if a.uoc_tinh:
        return uoc_tinh(d, a)

    # bỏ qua ảnh đã thu nhỏ -> chạy lại được, ngắt giữa chừng không mất gì
    viec = Queue()
    bo_qua = 0
    for r in d.itertuples(index=False):
        ra = a.ra / str(r.video_id) / f"{int(r.kf_n):03d}.jpg"
        if ra.exists():
            bo_qua += 1
            continue
        viec.put((Path(r.kf_path), ra))
        if a.n and viec.qsize() >= a.n:
            break
    tong = viec.qsize()
    print(f"cần sinh {tong:,}  (bỏ qua {bo_qua:,} đã có)")
    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi thật, `--uoc-tinh` để đo dung lượng)")
        return
    if tong == 0:
        print("Không còn gì để sinh.")
    else:
        khoa = threading.Lock()
        dem = {"xong": 0, "hong": 0, "byte": 0}
        t0 = time.perf_counter()

        def chay():
            while True:
                try:
                    goc, ra = viec.get_nowait()
                except Empty:
                    return
                try:
                    b = thu_nho(goc, a.rong, a.chat)
                    ra.parent.mkdir(parents=True, exist_ok=True)
                    ra.write_bytes(b)
                except Exception as e:
                    with khoa:
                        dem["hong"] += 1
                        if dem["hong"] <= 5:
                            print(f"  hỏng {goc.name}: {str(e)[:60]}", flush=True)
                    continue
                with khoa:
                    dem["xong"] += 1
                    dem["byte"] += len(b)
                    if dem["xong"] % 2000 == 0:
                        t = time.perf_counter() - t0
                        con = (tong - dem["xong"]) * t / dem["xong"] / 60
                        print(f"  {dem['xong']:,}/{tong:,}  "
                              f"{dem['xong'] / t:,.0f} ảnh/giây  còn ~{con:.0f} phút",
                              flush=True)

        ts = [threading.Thread(target=chay, daemon=True) for _ in range(a.luong)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        giay = time.perf_counter() - t0
        print(f"\nXong {dem['xong']:,} ảnh trong {giay / 60:.1f} phút  "
              f"({dem['byte'] / 1024 ** 3:.2f} GB)"
              + (f"  |  {dem['hong']} hỏng" if dem["hong"] else ""))

    if a.nen:
        import shutil
        z = shutil.make_archive(str(a.ra), "zip", str(a.ra))
        print(f"\n✅ {z}  ({Path(z).stat().st_size / 1024 ** 3:.2f} GB)")


if __name__ == "__main__":
    main()
