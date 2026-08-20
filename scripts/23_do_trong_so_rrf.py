"""
23_do_trong_so_rrf.py — Trọng số nào đúng cho RRF(objects, OCR)?

VÌ SAO CÓ SCRIPT NÀY — MỘT PHÁT HIỆN LÚC DỰNG LẠI BÀI NỘP
==========================================================

Dựng lại bộ đề mẫu bằng lệnh trong sổ tay ra **19/24 file khác** bản nộp đã ăn
2,6 điểm trên leaderboard. Truy ra: bản đã nộp chạy với `--trong-so-phu 1.0`,
còn **mặc định trong code là 0,3**. Tức chạy bằng mặc định hôm nay là nộp một
bài KHÁC bài đã ghi điểm, mà không có gì báo.

Con số 0,3 đến từ A14.2, và ở đó nó ĐÚNG: A14.2 đo cảnh **một kênh mạnh cộng
một kênh yếu** (SigLIP2 0,3258 với objects 0,0412 — chênh 8 lần), nên phải dìm
kênh yếu xuống kẻo nó pha loãng.

Nhưng cấu hình model-free thì ngược hẳn: **hai kênh ngang nhau** —

    kênh 4 objects   0,0412
    kênh 3 OCR       0,0420

Dìm một trong hai xuống 0,3 là vứt bỏ nửa bằng chứng. Đó là giả thuyết; script
này đo.

    python scripts/23_do_trong_so_rrf.py

⚠️ Đây là câu hỏi VỀ TRỌNG SỐ, không phải về việc có hợp nhất hay không —
việc đó A14.2 đã trả lời (có, khi hai kênh cùng tầm). Và trọng số là siêu tham
số: dò nhiều mức trên cùng một tập dev thì mức thắng có thể chỉ là mức **hợp
với nhiễu của tập này nhất**. Nên đọc kèm thắng–thua–hòa, và chỉ đổi mặc định
nếu chênh lệch vượt ngưỡng nhiễu.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import MOC_DUNG_SAI, bao_cao_do_nhay   # noqa: E402
from objects import KenhObjects                       # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)
    k4 = KenhObjects(a.index, master)
    p = a.index / "ocr_asr.parquet"
    if not p.exists():
        raise SystemExit(f"Chưa có {p} — cần kênh 3.")
    k3 = KenhVanBan.tu_bang_khung(master, pd.read_parquet(p),
                                  cot="text", ten="ocr_asr")

    print(f"{len(cau)} câu | kênh 3 có {len(k3):,} khung có chữ | k={a.k}\n")

    nho4, nho3 = {}, {}

    def f4(c):
        if c.id not in nho4:
            nho4[c.id] = k4.tim(c.cau_hoi, k=a.k)
        return nho4[c.id]

    def f3(c):
        if c.id not in nho3:
            nho3[c.id] = k3.tim(c.cau_hoi, k=a.k)
        return nho3[c.id]

    # Mốc nền là CẤU HÌNH ĐÃ NỘP THẬT (trọng số 1,0), không phải mặc định của
    # code. Câu hỏi cần trả lời là "có nên rời khỏi cấu hình đã ăn 2,6 điểm
    # không", nên mốc phải là chính nó.
    cau_hinh = {"trọng số 1,0 : 1,0  (ĐÃ NỘP)":
                lambda c: hop_nhat([f4(c), f3(c)], trong_so=[1.0, 1.0])}
    for w in (0.5, 0.3, 0.1):
        cau_hinh[f"trọng số 1,0 : {w}  (OCR bị dìm)"] = (
            lambda w: lambda c: hop_nhat([f4(c), f3(c)],
                                         trong_so=[1.0, w]))(w)
    cau_hinh["chỉ objects"] = f4
    cau_hinh["chỉ OCR"] = f3

    print(bao_cao_do_nhay(cau, cau_hinh, master, MOC_DUNG_SAI, gioi_han=a.k))


if __name__ == "__main__":
    main()
