"""
71_do_kenh5_caption.py — Kênh 5 (BM25 trên caption) có kéo đáp án lên hạng không?

    python scripts/71_do_kenh5_caption.py

A90 — ĐÃ ĐỦ 12/12 PHẦN: caption phủ 177.321 ảnh / 873 video = 100,0% kho.
Từ đây bể ứng viên KHÔNG còn bị khoá nhỏ nữa, nên điểm tuyệt đối so được với
các mục khác. Script tự đọc độ phủ thật từ `caption.parquet` và chỉ in cảnh báo
"khoá bể" khi độ phủ dưới 99,9%.

⚠️ VÌ SAO PHẢI ĐO LẠI DÙ A73 ĐÃ BÁC

A73 kết luận ❌ đảo dấu, nhưng đo ở độ phủ 76% VÀ trên 68 câu có 16 câu nhãn
hái từ chính hệ thống (A89). Cả hai điều kiện nay đã đổi, nên phải chạy lại
trên `--file dev/tap_de_that.jsonl` (52 câu nhãn sạch).

KẾT QUẢ (A90): KHÔNG LẬT, VÀ MẠNH HƠN. w=1,0 đi từ 🟡 sang ✅ ổn định TỆ HƠN
(−0,0596, thua 24/52). Dãy ba điểm đo của kênh 5 đứng một mình:

    độ phủ  5,9% -> 0,3904   (A59)
    độ phủ 76,0% -> 0,2625   (A73)
    độ phủ  100% -> 0,1615   (A90)

Càng phủ đủ càng thấp, ĐƠN ĐIỆU, ba trên ba. Đây là cơ chế A21 ở dạng thuần
khiết: bể càng nhỏ so với kho, con số càng nói về BỂ chứ không về KÊNH. Ở 5,9%
thì bể gần như chỉ gồm video CHỨA đáp án, nên kênh 5 chỉ phải chọn khung trong
một tập đã được lọc sẵn hộ.

CÁCH ĐO KHI ĐỘ PHỦ CHƯA ĐỦ: KHOÁ BỂ ỨNG VIÊN

Ép MỌI kênh chỉ được đề xuất trong các video có caption. Tất cả cùng nhìn một
vũ trụ nên so sánh công bằng — nhưng công bằng KHÔNG có nghĩa là đại diện, và
dãy ba số ở trên là bằng chứng cho đúng câu đó.
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
    # A89: bỏ `tap_de_thi_thu` khỏi mặc định — 20 câu đó mang nhãn hái từ
    # chính đầu ra hệ thống, thiên vị CÓ CHIỀU chống lại mọi cấu hình mới.
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl"])
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
    # Ở 100% độ phủ thì cảnh báo "bể bị khoá" thành vô nghĩa — và một cảnh báo
    # luôn hiện là một cảnh báo không ai còn đọc.
    if be.mean() < 0.999:
        print("\n⚠️ Điểm tuyệt đối ở đây KHÔNG so được với các mục khác — bể "
              f"chỉ còn {be.mean() * 100:.1f}% kho. Chỉ đọc HIỆU giữa các dòng.")
    else:
        print("\n✅ Bể phủ TRỌN kho — điểm tuyệt đối ở đây so được với các mục "
              "khác dùng cùng cỡ bể.")


if __name__ == "__main__":
    main()
