"""
26_do_don_cuc_trake.py — Chốt chống dồn cục của TRAKE: xét TỔNG hay xét TỪNG CẶP?

`run.dung_trake` có một chốt: nếu N khung dồn hết vào một chỗ thì bỏ đi, rải đều
trên khoảng frame của video. Bản đầu xét **tổng độ trải** `khung[-1]-khung[0]`,
nên **không bắt được dồn cục MỘT PHẦN** — ba sự kiện đầu trùng một khung còn sự
kiện cuối ở xa thì tổng độ trải vẫn lớn, chốt không nổ.

Đo trên chính bài nộp đã ăn 3,8 điểm:

    query-p1-18-trake   47/100 dòng có cặp sự kiện liền kề cách nhau < 100 frame
    query-p1-4-trake    33/100
    query-p1-16-trake    1/100

Ví dụ thật: `L23_V013,0,1,2,2298` — ba sự kiện cách nhau 0,03 giây.

    python scripts/26_do_don_cuc_trake.py

ĐO Ở TẦNG LẮP RÁP, KHÔNG PHẢI TẦNG KÊNH
========================================

Dùng **kênh 3 (OCR+ASR)** làm nguồn ứng viên vì nó chạy không cần model — máy
7,7 GB không nạp nổi SigLIP2. Điều đó KHÔNG làm hỏng phép so: cả ba biến thể ăn
**cùng một danh sách ứng viên**, chỉ khác nhau ở khâu lắp ráp — đúng thứ đang
cần so. Con số tuyệt đối thì thấp hơn thực chiến (kênh 1 mạnh gấp 2,8 lần), nên
chỉ đọc phần SO SÁNH.

Chấm bằng `diem_trake_bai_nop`: vị trí i chỉ được so với sự kiện i, đúng cách
BTC chấm bài nộp.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import MOC_DUNG_SAI, diem_trake_bai_nop, no_cua_so  # noqa: E402
from dense import KenhAnhCache                        # noqa: E402


def bang_tra_nguoc(master) -> dict:
    """`(video_id, frame_idx) -> [row_id, ...]` — A5.7: không phải song ánh."""
    d = defaultdict(list)
    for r in master.itertuples():
        d[(r.video_id, int(r.frame_idx))].append(int(r.row_id))
    return d


def main():
    ap = argparse.ArgumentParser(description="do chot chong don cuc cua TRAKE")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--cache", type=Path, default=None,
                    help="file .npz vector truy vấn đã mã hoá sẵn "
                         "(scripts/25_ma_hoa_truy_van.py) — dùng ỨNG VIÊN "
                         "KÊNH 1 (SigLIP2) thay vì kênh 3. Kênh 3 một mình "
                         "không tìm ra sự kiện TRAKE nào (điểm 0.0000 cả ba "
                         "biến thể), nên phép so chỉ có ý nghĩa với kênh 1")
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    tra = bang_tra_nguoc(master)
    cau = [c for c in tap_dev.doc() if c.loai == "TRAKE"]
    if not cau:
        raise SystemExit("Tập dev chưa có câu TRAKE nào.")

    if a.cache:
        print(f"Ứng viên từ kênh 1 (SigLIP2), cache {a.cache} — KHÔNG nạp model")
        k = KenhAnhCache(str(a.index), a.cache, matrix=a.matrix)
        thieu = k.co_du([sk for c in cau for sk in R.tach_su_kien(c.cau_hoi)])
        if thieu:
            raise SystemExit(
                f"{len(thieu)} mệnh đề sự kiện chưa có trong cache, ví dụ:\n"
                f"  {thieu[0][:100]!r}\n"
                f"Mã hoá thêm: python scripts/25_ma_hoa_truy_van.py "
                f"--tap-dev --gop")
    else:
        p = a.index / "ocr_asr.parquet"
        if not p.exists():
            raise SystemExit(f"Chưa có {p}")
        k = KenhVanBan.tu_bang_khung(master, pd.read_parquet(p),
                                      cot="text", ten="ocr_asr")
        print(f"Ứng viên từ kênh 3 ({len(k):,} khung có chữ)")
    print(f"{len(cau)} câu TRAKE")
    print("⚠️  n nhỏ — đọc thắng-thua-hoà, đừng đọc mỗi điểm trung bình\n")

    # Ứng viên dựng MỘT LẦN, dùng chung cho cả ba biến thể — nếu dựng lại theo
    # từng biến thể thì khác biệt đo được có thể đến từ nhiễu của kênh.
    kho = {}
    for c in cau:
        if a.cache:
            kho[c.id] = [k.tim(sk, k=a.k) for sk in R.tach_su_kien(c.cau_hoi)]
        else:
            kho[c.id] = [k.tim(R.tach_truy_van(sk), k=a.k)
                         for sk in R.tach_su_kien(c.cau_hoi)]

    for dung_sai in MOC_DUNG_SAI:
        print(f"=== dung sai ±{dung_sai}s " + "=" * 50)
        diem, dem_don = {}, {}
        for cach in ("tong", "cap", "khong"):
            tong, don = 0.0, 0
            for c in cau:
                dong = R.dung_trake(kho[c.id], master, so_dong=a.k,
                                    don_cuc=cach)
                # dòng nộp -> row_id để chấm; A5.7 nên một frame_idx có thể ứng
                # với nhiều row_id, lấy cái đầu tiên có thật
                cac_dong = []
                for d in dong:
                    r = [tra.get((d.video_id, int(f)), [None])[0]
                         for f in d.frame_idxs]
                    if all(x is not None for x in r):
                        cac_dong.append(r)
                    don += any(b - x < R.DON_NHAU
                               for x, b in zip(d.frame_idxs, d.frame_idxs[1:]))
                dung = [no_cua_so(rs, master, dung_sai) for rs in c.row_id_dung]
                tong += diem_trake_bai_nop(cac_dong, dung, a.k)
            diem[cach] = tong / len(cau)
            dem_don[cach] = don
        for cach in ("tong", "cap", "khong"):
            nhan = {"tong": "xét TỔNG độ trải (bản đầu)",
                    "cap": "xét TỪNG CẶP liền kề (mới)",
                    "khong": "tắt hẳn việc rải đều"}[cach]
            print(f"  {nhan:34} điểm {diem[cach]:.4f} | "
                  f"{dem_don[cach]:>5} dòng còn dồn cục")
        print()


if __name__ == "__main__":
    main()
