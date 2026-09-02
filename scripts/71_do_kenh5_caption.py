"""
71_do_kenh5_caption.py — Kênh 5 (BM25 trên caption) có kéo đáp án lên hạng không?

    python scripts/71_do_kenh5_caption.py

⚠️ VÌ SAO KHÔNG ĐO THẲNG NHƯ CÁC KÊNH KHÁC

`caption.parquet` hiện chỉ phủ **47 video mà tập đề thật đụng tới** (10.488 ảnh
= 5,9% kho) — cố ý, để biết kênh này có đáng 103 giờ GPU cho cả kho không.
Nay đã có 9/12 phần (**134.708 ảnh / 663 video = 76% kho**); script tự đọc
độ phủ thật từ `caption.parquet`.

⚠️ A73: độ phủ tăng 13 lần thì đóng góp của kênh 5 **biến mất** (+0,0106 ->
+0,0022 rồi đảo dấu). Con số cũ đẹp là vì bể quá nhỏ, không phải vì kênh tốt.

Nhưng dựng BM25 trên đúng ngần ấy thì kênh 5 **chỉ có thể đề xuất khung từ
chính những video chứa đáp án**. Nó sẽ cho ra con số đẹp rực rỡ và hoàn toàn vô
nghĩa — đúng cơ chế A21 đo được mức tăng ẢO 0,400 → 0,840.

CÁCH ĐO ĐÚNG: KHOÁ BỂ ỨNG VIÊN

Ép **mọi kênh** chỉ được đề xuất trong 47 video đó. Tất cả cùng nhìn một vũ trụ,
nên so sánh công bằng. Câu hỏi trở thành:

    "Khi video đúng đã nằm trong bể, caption có kéo đáp án lên hạng cao hơn
     không?"

Hợp lệ, và trả lời đúng vấn đề A54: R@20 = 0,61 nhưng R@1 = 0,20.

⚠️ ĐIỂM Ở ĐÂY KHÔNG SO ĐƯỢC VỚI ĐIỂM Ở CÁC MỤC KHÁC. Bể bị khoá nhỏ hơn kho nên
mọi cấu hình đều cao vọt. Chỉ đọc HIỆU giữa các dòng, đừng trích con số tuyệt
đối ra ngoài script này.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
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
W5 = (0.25, 0.5, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--caption", default=GOC / "index" / "caption.parquet",
                    type=Path)
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cap = pd.read_parquet(a.caption)
    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)

    # Bể ứng viên = mọi keyframe của các video CÓ caption.
    vid_co = set(master.video_id.iloc[cap.row_id.values])
    be = master.video_id.isin(vid_co).values
    print(f"caption: {len(cap):,} ảnh / {len(vid_co)} video")
    print(f"bể ứng viên khoá còn {int(be.sum()):,}/{len(master):,} keyframe "
          f"({be.mean() * 100:.1f}% kho)\n")

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    k5 = KenhVanBan.tu_bang_khung(master, cap, cot="caption", ten="caption")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    con = [c for c in giu
           if any(be[r] for r in (c.row_id_dung if c.loai != "TRAKE"
                                  else [x for b in c.row_id_dung for x in b]))]
    print(f"{'+'.join(f.stem for f in a.file)}: {len(con)}/{len(giu)} câu có đáp án trong bể\n")

    tho = {}

    def nen(c):
        if c.id not in tho:
            anh = hop_nhat([k1.tim(m, k=a.be, be=be)
                            for m in R.tach_truy_van(c.cau_hoi)])
            tho[c.id] = (anh, k3.tim(c.cau_hoi, k=a.be, be=be),
                         k5.tim(c.cau_hoi, k=a.be, be=be))
        return tho[c.id]

    def _nho(f):
        n = {}

        def g(c):
            if c.id not in n:
                n[c.id] = f(c)[:100]
            return n[c.id]
        return g

    cau_hinh = {
        "1. mốc: ảnh + kênh 3 (khoá bể)":
            _nho(lambda c: hop_nhat(list(nen(c)[:2]), trong_so=[1.0, W3])),
        "2. chỉ kênh 5 (chẩn đoán)": _nho(lambda c: nen(c)[2]),
        "3. chỉ kênh 1 (chẩn đoán)": _nho(lambda c: nen(c)[0]),
    }
    for w in W5:
        cau_hinh[f"4. + kênh 5 ({w:g})"] = _nho(
            (lambda w: lambda c: hop_nhat(list(nen(c)), trong_so=[1.0, W3, w]))(w))
    cau_hinh["5. kênh 5 THAY kênh 3"] = _nho(
        lambda c: hop_nhat([nen(c)[0], nen(c)[2]], trong_so=[1.0, W3]))

    print(bao_cao_do_nhay(giu, cau_hinh, master))
    print("\n⚠️ Điểm tuyệt đối ở đây KHÔNG so được với các mục khác — bể chỉ "
          f"còn {be.mean() * 100:.1f}% kho. Chỉ đọc HIỆU giữa các dòng.")


if __name__ == "__main__":
    main()
