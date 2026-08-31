"""
52_do_trong_so_gopt.py — Trọng số nào cho RRF(gopt, OCR)?

    python scripts/52_do_trong_so_gopt.py

A47 chốt `RRF(gopt, OCR)` làm cấu hình mặc định, nhưng **cả bảng đó dùng 1:1
và chưa ai đo trọng số**. A45 cho thấy trọng số đổi kết quả đáng kể, nên đây là
việc còn lại rẻ nhất mà có lãi.

CHỈ ĐỔI MỘT THỨ

Mốc nền là `RRF(gopt, OCR)` tỉ lệ **1,0 : 1,0** — đúng cấu hình đang chạy. Mọi
dòng khác chỉ khác nó ở **một con số**: trọng số của kênh OCR. Không đổi model,
không đổi bể, không đổi tập câu.

`trong_so=[1.0, w]`: kênh đầu là gopt, kênh sau là OCR.
`w < 1` nghĩa là tin gopt hơn; `w > 1` nghĩa là tin OCR hơn.

⚠️ Đọc `bao_cao_do_nhay`. Với ~50 câu, một trọng số "hơn 0,004" là nhiễu, không
phải phát hiện. Chỉ nhận cái nào ✅ ỔN ĐỊNH.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache, be_chung              # noqa: E402
from rrf import hop_nhat                              # noqa: E402

TRONG_SO = (0.25, 0.5, 0.75, 1.5, 2.0, 3.0)


def nho(k, **kw):
    """Bọc kênh, nhớ kết quả — `bao_cao_do_nhay` chấm ở hai mức và mọi cấu
    hình dùng chung hai kênh, không nhớ thì chạy lại hàng chục lần mỗi câu."""
    cache = {}

    def f(c):
        if c.id not in cache:
            cache[c.id] = k.tim(c.cau_hoi, k=100, **kw)
        return cache[c.id]
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k_si = KenhAnhCache(str(a.index), str(a.index / "truy_van.npz"),
                        matrix="clip_siglip2.npy")
    k_go = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                        matrix="clip_gopt.npy")
    be = be_chung(k_si, k_go)
    k_ocr = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    # Loại đúng những câu A47 đã loại, để con số so được với A47.
    giu = [c for c in cau
           if not k_si.co_du([c.cau_hoi]) and not k_go.co_du([c.cau_hoi])]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu"
          f" | bể chung {int(be.sum()):,}\n")

    f_go, f_ocr = nho(k_go, be=be), nho(k_ocr)

    cau_hinh = {"1,0 : 1,0  MỐC (A47)":
                lambda c: hop_nhat([f_go(c), f_ocr(c)])}
    for w in TRONG_SO:
        cau_hinh[f"1,0 : {w:<4g} gopt:OCR"] = (
            lambda w: lambda c: hop_nhat([f_go(c), f_ocr(c)],
                                         trong_so=[1.0, w]))(w)
    cau_hinh["gopt một mình"] = f_go

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
