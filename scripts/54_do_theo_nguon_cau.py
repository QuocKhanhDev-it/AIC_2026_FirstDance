"""
54_do_theo_nguon_cau.py — gopt trần hay RRF(gopt, OCR)? Tách CỠ MẪU khỏi PHÂN BỐ.

    python scripts/54_do_theo_nguon_cau.py

VẤN ĐỀ

Hai tập nói ngược nhau:

  tap_de_that (50 câu)   gopt 0,3160  <  RRF(gopt,OCR) 0,3440
  tap_dev    (323 câu)   gopt 0,4657  >  RRF(gopt,OCR) 0,4408

Không chọn được bằng cách nhìn hai bảng: chúng khác nhau ở **hai** thứ cùng
lúc — cỡ mẫu (50 so với 323) và nguồn câu hỏi (đề thật so với tự soạn). Đúng
cái lỗi "đổi hai thứ rồi quy công cho nhầm cái".

CÁCH TÁCH

Chia `tap_dev` theo NGUỒN, đo riêng từng nhóm, cùng một cấu hình:

  đề thật          52 câu   BTC viết
  mới sát đề thật  63 câu   tự soạn, cố ý khớp phân bố đề thật
  tự soạn cũ      208 câu   tự soạn từ trước

Đọc kết quả:

  * gopt thắng ở CẢ BA  -> cỡ mẫu là lý do; tin bảng 323 câu, bỏ OCR.
  * gopt chỉ thắng ở nhóm tự soạn -> nguồn câu là lý do; tin đề thật, giữ RRF.
  * lẫn lộn -> chưa kết luận được, đừng đổi mặc định.

⚠️ Nhóm "đề thật" chỉ 52 câu nên ngưỡng nhiễu của nó rộng. Đọc **hướng** và
**thắng-thua-hoà**, đừng đọc mỗi hiệu số.
"""

import argparse
import json
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


def _id(f: Path) -> set:
    return {json.loads(l)["id"]
            for l in f.read_text("utf-8").splitlines() if l.strip()}


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
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(GOC / "dev" / "tap_dev.jsonl")

    that = _id(GOC / "dev" / "tap_de_that.jsonl")
    moi = _id(GOC / "dev" / "tap_dev_bonus-30-8.jsonl")

    def nguon(c):
        return ("đề thật" if c.id in that
                else "mới sát đề thật" if c.id in moi else "tự soạn cũ")

    k_si = KenhAnhCache(str(a.index), str(a.index / "truy_van.npz"),
                        matrix="clip_siglip2.npy")
    k_go = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                        matrix="clip_gopt.npy")
    be = be_chung(k_si, k_go)
    k_ocr = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    f_go, f_ocr = nho(k_go, be=be), nho(k_ocr)
    cau_hinh = {
        "gopt trần  MỐC": f_go,
        "RRF(gopt, OCR)": lambda c: hop_nhat([f_go(c), f_ocr(c)]),
        "RRF 1:0,75": lambda c: hop_nhat([f_go(c), f_ocr(c)],
                                         trong_so=[1.0, 0.75]),
    }

    for ten in ("đề thật", "mới sát đề thật", "tự soạn cũ"):
        nhom = [c for c in cau if nguon(c) == ten]
        print("\n" + "=" * 74)
        print(f"NHÓM: {ten}  ({len(nhom)} câu)")
        print("=" * 74)
        print(bao_cao_do_nhay(nhom, cau_hinh, master))


if __name__ == "__main__":
    main()
