"""
53_do_trong_so_va_k.py — SigLIP2 còn giá trị nào không, và hằng số k của RRF.

    python scripts/53_do_trong_so_va_k.py --phan trong-so
    python scripts/53_do_trong_so_va_k.py --phan k

HAI CÂU RIÊNG BIỆT, ĐO RIÊNG

A47 thấy thêm SigLIP2 vào `RRF(gopt, OCR)` làm điểm tụt, nhưng chỉ 🟡 YẾU —
chưa đủ để tuyên bố. Có thể vì trọng số 1:1 quá nặng tay: nếu SigLIP2 chỉ đóng
vai "gợi ý phụ" cứu vài ca gopt bỏ sót thì phải cho nó trọng số nhỏ.

  --phan trong-so : gopt : SigLIP2 ở nhiều tỉ lệ, có cả 1:0 (bỏ hẳn SigLIP2)
  --phan k        : hằng số k của RRF, giữ nguyên cấu hình tốt nhất

k CÓ Ý NGHĨA GÌ

RRF cộng `1/(k + hạng)`. k lớn làm khoảng cách giữa hạng 1 và hạng 50 co lại —
kênh yếu khó kéo kết quả sai lên đầu hơn. k nhỏ thì chỉ hạng siêu cao mới có
trọng lượng, kênh yếu gần như bị triệt tiêu.

⚠️ ĐỔI MỘT THỨ MỖI LẦN. Hai phần chạy riêng chính vì thế — dò trọng số và dò k
cùng lúc là không biết công thuộc về cái nào.
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

# gopt : SigLIP2 — nhỏ dần, tức SigLIP2 càng lúc càng chỉ là gợi ý phụ
TS_SIGLIP2 = (1.0, 0.5, 0.33, 0.25, 0.2)
HANG_SO_K = (20, 30, 100, 120, 200)


def nho(k, **kw):
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
    ap.add_argument("--phan", choices=("trong-so", "k"), default="trong-so")
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

    giu = [c for c in cau
           if not k_si.co_du([c.cau_hoi]) and not k_go.co_du([c.cau_hoi])]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu"
          f" | bể chung {int(be.sum()):,}\n")

    f_go, f_si, f_ocr = nho(k_go, be=be), nho(k_si, be=be), nho(k_ocr)

    if a.phan == "trong-so":
        # Mốc: cấu hình đang chạy, KHÔNG có SigLIP2.
        cau_hinh = {"gopt+OCR, KHÔNG SigLIP2  MỐC":
                    lambda c: hop_nhat([f_go(c), f_ocr(c)])}
        for w in TS_SIGLIP2:
            cau_hinh[f"+ SigLIP2 trọng số {w:<4g}"] = (
                lambda w: lambda c: hop_nhat(
                    [f_go(c), f_ocr(c), f_si(c)], trong_so=[1.0, 1.0, w]))(w)
    else:
        cau_hinh = {"k = 60  MỐC (mặc định)":
                    lambda c: hop_nhat([f_go(c), f_ocr(c)])}
        for kk in HANG_SO_K:
            cau_hinh[f"k = {kk}"] = (
                lambda kk: lambda c: hop_nhat([f_go(c), f_ocr(c)], k=kk))(kk)

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
