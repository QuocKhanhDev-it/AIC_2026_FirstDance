"""
59_do_loc_menh_de.py — Lọc bớt mệnh đề trước khi đưa vào RRF có lãi không?

    python scripts/59_do_loc_menh_de.py

Ý TƯỞNG ĐEM ĐO

A51 cho mỗi mệnh đề một phiếu ngang nhau trong RRF. Nếu trong đó có mệnh đề
quá ngắn hoặc chỉ là câu dẫn ("Bốn khoảnh khắc sau xuất hiện lần lượt."), nó
vẫn kéo về 100 ứng viên và vẫn được một phiếu — nhiễu thuần, mà lại có trọng
lượng bằng mệnh đề tả đúng cảnh cần tìm.

CHI PHÍ BẰNG KHÔNG. Lọc chỉ BỎ BỚT mệnh đề, không sinh chuỗi mới, nên mọi thứ
cần mã hoá đều đã có sẵn trong cache.

BỐN CÁCH LỌC

  1. bỏ mệnh đề < 4 từ
  2. bỏ mệnh đề < 8 từ
  3. bỏ mệnh đề KHÔNG có danh từ/tính từ nội dung — xấp xỉ bằng "toàn từ chức
     năng và từ dẫn" (`TU_DAN`)
  4. giữ N mệnh đề DÀI NHẤT (dài ~ nhiều chi tiết thị giác hơn)

⚠️ LỌC LÀ CON DAO HAI LƯỠI. Bỏ nhầm một mệnh đề đặc trưng thì mất đúng cái
mệnh đề duy nhất tìm ra khung đáp án. Nên bảng dưới in kèm số mệnh đề bị bỏ —
một cách lọc "thắng" nhưng chẳng bỏ gì thì chỉ là nhiễu đo.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5                                              # trọng số kênh 3 (A52)

# Câu dẫn của đề TRAKE/KIS: nói về cấu trúc câu hỏi, không tả cảnh nào cả.
TU_DAN = ("khoảnh khắc sau", "các cảnh", "cảnh sau", "xuất hiện lần lượt",
          "liên tiếp nhau", "tìm các sự kiện", "đoạn video", "video về",
          "trong video", "sau đây", "gồm các")


def loc_ngan(md, n):
    return [m for m in md if len(m.split()) >= n] or md


def loc_cau_dan(md):
    """Bỏ mệnh đề mà phần lớn nội dung là câu dẫn cấu trúc."""
    ra = [m for m in md
          if not any(t in m.lower() for t in TU_DAN) or len(m.split()) > 25]
    return ra or md


def giu_dai_nhat(md, n):
    return sorted(md, key=lambda m: -len(m.split()))[:n] or md


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu")

    # Phân bố độ dài mệnh đề — biết trước có gì để lọc hay không.
    dai = sorted(len(m.split()) for c in giu for m in R.tach_truy_van(c.cau_hoi))
    print(f"mệnh đề: {len(dai)} cái | từ/mệnh đề: min {dai[0]}, "
          f"trung vị {dai[len(dai) // 2]}, max {dai[-1]} | "
          f"< 4 từ: {sum(1 for x in dai if x < 4)}, "
          f"< 8 từ: {sum(1 for x in dai if x < 8)}")

    cach = {
        "1. mốc: mọi mệnh đề (run.py)": lambda md: md,
        "2. bỏ mệnh đề < 4 từ": lambda md: loc_ngan(md, 4),
        "3. bỏ mệnh đề < 8 từ": lambda md: loc_ngan(md, 8),
        "4. bỏ câu dẫn cấu trúc": loc_cau_dan,
        "5. giữ 3 mệnh đề dài nhất": lambda md: giu_dai_nhat(md, 3),
        "6. giữ 2 mệnh đề dài nhất": lambda md: giu_dai_nhat(md, 2),
    }

    ocr, cau_hinh = {}, {}
    for ten, f in cach.items():
        bo = sum(len(R.tach_truy_van(c.cau_hoi)) - len(f(R.tach_truy_van(c.cau_hoi)))
                 for c in giu)
        print(f"   {ten:<32} bỏ {bo} mệnh đề")

        def lam(f):
            nho = {}

            def g(c):
                if c.id not in nho:
                    md = f(R.tach_truy_van(c.cau_hoi))
                    anh = hop_nhat([k1.tim(m, k=100) for m in md])
                    if c.id not in ocr:
                        ocr[c.id] = k3.tim(c.cau_hoi, k=100)
                    nho[c.id] = hop_nhat([anh, ocr[c.id]], trong_so=[1.0, W3])
                return nho[c.id]
            return g
        cau_hinh[ten] = lam(f)

    print()
    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
