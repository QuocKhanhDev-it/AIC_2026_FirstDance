"""
65_do_co_be.py — Cỡ bể ứng viên có nên là mặc định lớn hơn 100 không?

    python scripts/65_do_co_be.py

A54 đo trần theo ba cỡ bể và thấy ĐIỂM THẬT cũng nhích theo: 0,5173 (bể 100)
-> 0,5202 (300) -> 0,5317 (1000). Nhưng đó là ba bảng riêng, chưa so theo cặp,
nên chưa biết chênh lệch đó có vượt nhiễu hay không. Script này so.

VÌ SAO NỚI BỂ LẠI ĐỔI ĐƯỢC ĐIỂM

Bài nộp vẫn 100 dòng — giới hạn của BTC là số DÒNG NỘP, không phải cỡ bể. Cái
đổi là **đầu vào của RRF**: mỗi kênh trả về nhiều ứng viên hơn, nên hai kênh có
thêm chỗ để đồng thuận. Một khung mà kênh 1 xếp hạng 250 và kênh 3 xếp hạng 40
hoàn toàn không tồn tại trong bể 100 — cả hai kênh đều "biết" nó, mà hợp nhất
thì không thấy.

CHI PHÍ: chỉ là `k` truyền cho mỗi kênh. Không dữ liệu mới, không model mới.
Bể 1000 tốn thêm ~3 lần thời gian truy hồi mỗi câu (vẫn dưới một giây).
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

W3 = 0.5
CO_BE = (100, 200, 300, 600, 1000)


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
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu\n")

    def lam(be):
        nho = {}

        def g(c):
            if c.id not in nho:
                anh = hop_nhat([k1.tim(m, k=be) for m in R.tach_truy_van(c.cau_hoi)])
                # Bài nộp LUÔN 100 dòng — chỉ bể là khác.
                nho[c.id] = hop_nhat([anh, k3.tim(c.cau_hoi, k=be)],
                                     trong_so=[1.0, W3])[:100]
            return nho[c.id]
        return g

    cau_hinh = {f"bể {be}" + ("  ← MỐC (run.py)" if be == 100 else ""): lam(be)
                for be in CO_BE}
    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
