"""
51_do_tran_top100.py — Video đúng có lọt top-100 không? Trần của mọi hậu xử lý.

    python scripts/51_do_tran_top100.py --file dev/tap_de_that.jsonl

A46 đo con số này bằng SigLIP2 và ra **19/30 = 63%**, rồi kết luận "việc đáng
làm là kéo 63% lên, không phải xếp lại tinh vi hơn". A47 đổi kênh 1 sang gopt,
nên con số đó không còn đúng — phải đo lại trước khi ai đó trích nó.

VÌ SAO ĐO CẤP VIDEO CHỨ KHÔNG CẤP KHUNG

Xếp lại, mở rộng lân cận, tác tử VLM — mọi thứ hậu xử lý đều chỉ hoán vị hoặc
mở rộng quanh bể ứng viên có sẵn. Video đúng không lọt vào bể thì **không cách
nào cứu**. Đây là trần cứng, và nó là con số quyết định nên đầu tư vào đâu.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from dense import KenhAnhCache, be_chung              # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    vid = master.video_id.values
    cau = tap_dev.doc(a.file)

    k_si = KenhAnhCache(str(a.index), str(a.index / "truy_van.npz"),
                        matrix="clip_siglip2.npy")
    k_go = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                        matrix="clip_gopt.npy")
    be = be_chung(k_si, k_go)
    k_ocr = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau
           if not k_si.co_du([c.cau_hoi]) and not k_go.co_du([c.cau_hoi])]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu (bỏ câu thiếu cache)\n")

    def video_dung(c) -> set:
        r = c.row_id_dung
        p = [x for b in r for x in b] if isinstance(r[0], list) else r
        return {vid[x] for x in p}

    cau_hinh = {
        "SigLIP2 (A46 đo cái này)": lambda c: k_si.tim(c.cau_hoi, k=a.k, be=be),
        "gopt": lambda c: k_go.tim(c.cau_hoi, k=a.k, be=be),
        "RRF(SigLIP2, OCR) cũ": lambda c: hop_nhat(
            [k_si.tim(c.cau_hoi, k=a.k, be=be), k_ocr.tim(c.cau_hoi, k=a.k)]),
        "RRF(gopt, OCR) MỚI": lambda c: hop_nhat(
            [k_go.tim(c.cau_hoi, k=a.k, be=be), k_ocr.tim(c.cau_hoi, k=a.k)]),
    }

    print(f"{'cấu hình':<26}{'video đúng ∈ top-' + str(a.k):>20}{'hạng 1':>9}"
          f"{'top-5':>8}{'top-20':>8}")
    print("-" * 71)
    for ten, f in cau_hinh.items():
        lot = h1 = h5 = h20 = 0
        for c in giu:
            dung = video_dung(c)
            hang = next((i for i, x in enumerate(f(c)[:a.k], 1)
                         if vid[x.row_id] in dung), None)
            if hang:
                lot += 1
                h1 += hang == 1
                h5 += hang <= 5
                h20 += hang <= 20
        print(f"{ten:<26}{lot:>8}/{len(giu):<3} = {lot / len(giu) * 100:>4.0f}%"
              f"{h1:>9}{h5:>8}{h20:>8}")


if __name__ == "__main__":
    main()
