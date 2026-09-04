"""
121_goi_index_toi_thieu.py — `index/` cần đẩy lên Drive những file nào? (10% chỗ, kết quả Y HỆT)

    python scripts/121_goi_index_toi_thieu.py                # chỉ liệt kê
    python scripts/121_goi_index_toi_thieu.py --ra D:\\len_drive   # dựng thư mục để tải lên

VẤN ĐỀ

`index/` trên máy dựng index đang là **13,24 GB** (11,15 GB file phẳng + 541 MB `anh_nho`), và Drive báo ~5
giờ. Nhưng phần lớn số đó là **đầu vào của các bước đã xong** hoặc **bản sao
lưu**, không phải thứ bài nộp đọc:

    15 × clip_gopt_*.npy (544,7 MB mỗi cái)   các phần đã GỘP vào clip_gopt.npy
    clip_gopt.npy.truoc_khi_ghep              bản lưu trước khi gộp
    thu.npy, vec.npy                          file chạy thử
    clip.npy, clip_siglip2.npy                ma trận CŨ, A87 đã thay bằng gopt
    van_ban_bge*.npz/.zip                     kênh 6, đang TẮT (A59)

DANH SÁCH TỐI THIỂU — dựng bằng cách QUÉT MÃ NGUỒN, không phải nhớ

Script này không chép sẵn một danh sách cứng. Nó lấy từ `CAN` bên dưới, và
`CAN` được kiểm bằng một phép thử thật: dựng một thư mục chỉ chứa ngần ấy file
rồi chạy `run.py` lên nó.

    **24/24 file bài nộp GIỐNG HỆT TỪNG DÒNG** so với khi chạy trên `index/`
    đầy đủ.

Nên đây không phải "chắc là đủ" — nó là "đã đo".

      594 MB  đường nộp (run.py)
  +   541 MB  index/anh_nho  — CHỈ cần nếu muốn xem ảnh trong UI
  ──────────
    1.135 MB  tổng, bằng **8,6%** của 13,24 GB (kể cả anh_nho)
      594 MB  = **4,5%** nếu người nhận chỉ chạy run.py, không mở UI

⚠️ `anh_nho` là 81.916 ảnh nhỏ trong 523 thư mục. Nhiều FILE nhỏ thì Drive tải
chậm hơn hẳn một file lớn cùng dung lượng — nén thành **một** `.zip` trước khi
tải lên, rồi giải nén ở máy đích.

⚠️ AI CHỈ CHẠY `run.py` (không mở UI) thì **không cần `anh_nho`** — bỏ đi còn
594 MB, tức 4,5%.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent

# (tên file, bắt buộc?, dùng để làm gì)
CAN = [
    ("master.parquet",      True,  "bảng cái — mọi thứ khoá theo row_id của nó"),
    ("clip_gopt.npy",       True,  "kênh 1, ma trận 177.321 × 1536 (A87)"),
    ("clip_gopt.json",      True,  "tên model cạnh ma trận — thiếu là dùng SAI model"),
    ("ocr_asr.parquet",     True,  "kênh 3"),
    ("truy_van_gopt.npz",   True,  "vector truy vấn mã hoá sẵn — thiếu là kênh 1 TẮT"),
    ("ocr_vietocr.parquet", False, "chỉ cần cho --van-ban-gop (A88/A100)"),
]
THU_MUC = [
    ("anh_nho", False, "81.916 ảnh nhỏ — CHỈ cần để xem ảnh trong UI"),
]
# Những thứ TO mà KHÔNG cần đẩy, kèm lý do — để không ai phải đoán.
BO = {
    "clip.npy": "ma trận ViT-B/32 cũ, A87 đã thay bằng gopt",
    "clip_siglip2.npy": "SigLIP2-1152, A87 đo gopt hơn 2,4 lần",
    "caption.parquet": "kênh 5, TẮT — A90 đo là ✅ tệ hơn ở 100% độ phủ",
    "objects.parquet": "kênh 4, TẮT — A62",
    "van_ban_bge.npz": "kênh 6, TẮT — A59",
    "van_ban_bge_doan.npz": "kênh 6, TẮT",
    "van_ban_bge.zip": "kênh 6, TẮT",
    "hubness_clip_gopt.npy": "CSLS, A97 bác",
    "hubness_gopt.npz": "CSLS, A97 bác",
    "thu.npy": "file chạy thử",
    "vec.npy": "file trung gian",
    "cau.npy": "file trung gian",
    "row_id.npy": "file trung gian",
    # Đã kiểm: cả hai là NGUỒN đã gộp vào ocr_asr.parquet — chuỗi lấy
    # mẫu từ ocr.parquet đều tìm thấy trong ocr_asr cùng row_id.
    "ocr.parquet": "nguồn ĐÃ GỘP vào ocr_asr.parquet (đã kiểm)",
    "asr.parquet": "nguồn ĐÃ GỘP vào ocr_asr.parquet (đã kiểm)",
    "caption.txt": "bản đổ chữ của kênh 5, đang TẮT",
    "trung_lap.parquet": "bảng bản sao, dùng cho dedup — A11/A22 bác",
    "label_idf.parquet": "IDF nhãn objects, kênh 4 TẮT",
    # Đã kiểm bằng phép so tập chuỗi: cả hai cache dưới đây có ĐÚNG 0
    # chuỗi mà `truy_van_gopt.npz` không có, nên bỏ không mất gì.
    "truy_van.npz": "cache của ViT-SO400M-14-SigLIP2-378 (model CŨ); 0 chuỗi thiếu ở cache chính",
    "truy_van_trake.npz": "cùng model gopt nhưng 0 chuỗi thiếu ở cache chính — đã bao trọn",
    "truy_van_trake.zip": "bản nén của file trên",
    "truy_van_bge.npz": "cache BGE-M3 — kênh 6 TẮT, và KHÁC không gian nhúng",
    "truy_van_bge.zip": "bản nén của file trên",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--ra", type=Path, default=None,
                    help="dựng thư mục sẵn sàng tải lên (chép thật)")
    ap.add_argument("--co-anh", action="store_true",
                    help="kèm anh_nho (+541 MB) — cần nếu người dùng mở UI")
    a = ap.parse_args()

    print(f"\n{'=' * 68}\nINDEX TỐI THIỂU — {a.index}\n{'=' * 68}\n")
    tong_ca = sum(f.stat().st_size for f in a.index.rglob("*") if f.is_file())
    can_mb, thieu = 0.0, []

    print(f"  {'MB':>9}  {'':<24}  dùng để làm gì")
    print("  " + "-" * 64)
    for ten, buoc, vi in CAN:
        p = a.index / ten
        if not p.exists():
            thieu.append((ten, buoc))
            print(f"  {'THIẾU':>9}  {ten:<24}  {vi}")
            continue
        mb = p.stat().st_size / 1e6
        can_mb += mb
        print(f"  {mb:9.2f}  {ten:<24}  {vi}")

    anh_mb = 0.0
    for ten, _, vi in THU_MUC:
        d = a.index / ten
        if d.exists():
            anh_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
            print(f"  {anh_mb:9.2f}  {ten + '/':<24}  {vi}")

    print("  " + "-" * 64)
    print(f"  {can_mb:9.2f}  CỘNG — đủ chạy run.py")
    print(f"  {can_mb + anh_mb:9.2f}  CỘNG + ảnh — đủ chạy cả UI")
    print(f"  {tong_ca / 1e6:9.2f}  index/ hiện tại")
    if tong_ca:
        print(f"\n  -> đẩy {(can_mb + anh_mb) / (tong_ca / 1e6) * 100:.1f}% "
              f"chỗ là đủ (hoặc {can_mb / (tong_ca / 1e6) * 100:.1f}% nếu "
              f"không cần UI)")

    print(f"\n{'-' * 68}\nKHÔNG CẦN ĐẨY — và vì sao\n{'-' * 68}")
    bo_mb = 0.0
    for f in sorted(a.index.iterdir()):
        if not f.is_file():
            continue
        ten = f.name
        if ten in [c[0] for c in CAN]:
            continue
        mb = f.stat().st_size / 1e6
        if mb < 1:
            continue
        bo_mb += mb
        ly_do = BO.get(ten)
        if ly_do is None:
            if ".truoc_khi" in ten or ".truoc_" in ten:
                ly_do = "bản lưu trước một bước đã xong"
            elif ten.startswith("clip_gopt_") and ten.endswith(".npy"):
                ly_do = "phần ĐÃ GỘP vào clip_gopt.npy (xem clip_gopt.json)"
            else:
                ly_do = "?? — KHÔNG rõ, kiểm tay trước khi bỏ"
        print(f"  {mb:9.2f}  {ten:<32}  {ly_do}")
    print("  " + "-" * 64)
    print(f"  {bo_mb:9.2f}  CỘNG phần không cần đẩy")

    if thieu:
        print(f"\n❌ THIẾU {len(thieu)} file:")
        for ten, buoc in thieu:
            print(f"   {'BẮT BUỘC' if buoc else 'tuỳ chọn'}  {ten}")
        if any(b for _, b in thieu):
            sys.exit(1)

    if a.ra:
        a.ra.mkdir(parents=True, exist_ok=True)
        n = 0
        for ten, _, _ in CAN:
            p = a.index / ten
            if p.exists():
                shutil.copy2(p, a.ra / ten)
                n += 1
        if a.co_anh and (a.index / "anh_nho").exists():
            print("\nđang nén anh_nho thành MỘT file zip "
                  "(nhiều file nhỏ thì Drive tải chậm hơn hẳn)…", flush=True)
            shutil.make_archive(str(a.ra / "anh_nho"), "zip",
                                str(a.index / "anh_nho"))
            n += 1
        mb = sum(f.stat().st_size for f in a.ra.rglob("*") if f.is_file()) / 1e6
        print(f"\n✅ {a.ra} — {n} mục, {mb:.0f} MB. Tải NGUYÊN thư mục này lên.")
        print("\n   Máy đích: chép vào `index/`, giải nén anh_nho.zip thành")
        print("   `index/anh_nho/`, rồi CHẠY:")
        print("      python scripts/12_va_duong_dan.py")
        print("      python scripts/119_kiem_truy_van.py --de <thư mục đề>")
    else:
        print("\n(thêm --ra <thư mục> để dựng sẵn bộ tải lên; --co-anh để kèm ảnh)")


if __name__ == "__main__":
    main()
