"""
84_do_tuong_quan_kenh.py — Các kênh có nói ĐIỀU KHÁC NHAU không, hay lặp lại nhau?

    python scripts/84_do_tuong_quan_kenh.py

CÂU HỎI, VÀ VÌ SAO NÓ QUYẾT ĐỊNH MỘT KHOẢN GPU THẬT

A59 đo kênh 5 (caption) đứng một mình 0,3904 nhưng hợp nhất chỉ **+0,0106**.
Có đúng hai cách giải thích, và chúng dẫn tới hai quyết định trái ngược:

  (a) Kênh 5 YẾU  -> sinh caption cho cả kho cũng chỉ được ngần ấy. DỪNG.
  (b) Kênh 5 TRÙNG kênh 1 -> RRF không thấy nó "khác", nên không cộng hưởng.
      Caption nhiều hơn cũng không chữa được, vì cùng nhìn một tấm ảnh. DỪNG.
  (c) Kênh 5 KHÁC và MẠNH, chỉ là phủ có 5,9% kho -> ĐÁNG chạy tiếp 9 phần.

Điểm số không tách được ba khả năng đó. Độ trùng thì có.

⚠️ ĐO TRÙNG LẶP, KHÔNG ĐO ĐIỂM. Hai thước:

  * `chồng@k` — bao nhiêu phần trăm top-k của hai kênh là CÙNG row_id. Đây là
    đúng đại lượng RRF cần: RRF chỉ cộng hưởng khi hai kênh đề cử **cùng
    row_id** (A14 đo được 5/97 câu, và đó là lý do RRF thô thua).
  * `Spearman` — tương quan hạng trên phần giao. Cao = hai kênh xếp giống nhau
    ở chỗ chúng cùng thấy; thấp = chúng bất đồng.

Đọc bảng: **chồng@20 cao + Spearman cao = thừa**. **chồng@20 thấp = RRF không
có gì để cộng hưởng** (A14). Vùng đáng giá là ở giữa.

Bể ứng viên khoá theo `caption.parquet` như `71_do_kenh5_caption.py`, vì kênh 5
chỉ phủ một phần kho. Mọi kênh cùng nhìn một vũ trụ thì mới so được.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def spearman(a: dict, b: dict) -> float | None:
    """Tương quan hạng Spearman trên các row_id CẢ HAI kênh cùng đề cử."""
    chung = sorted(set(a) & set(b))
    if len(chung) < 5:
        return None
    x = np.array([a[r] for r in chung], float)
    y = np.array([b[r] for r in chung], float)
    x -= x.mean()
    y -= y.mean()
    m = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / m) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
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

    vid_co = set(master.video_id.iloc[cap.row_id.values])
    be = master.video_id.isin(vid_co).values
    print(f"caption: {len(cap):,} ảnh / {len(vid_co)} video")
    print(f"bể khoá còn {int(be.sum()):,}/{len(master):,} keyframe "
          f"({be.mean() * 100:.1f}% kho)\n")

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    k5 = KenhVanBan.tu_bang_khung(master, cap, cot="caption", ten="caption")

    giu = [c for c in cau
           if c.loai != "TRAKE" and not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)} câu đo được\n")

    ten = ["1 ảnh", "3 ocr+asr", "5 caption"]
    chong = {p: {k: [] for k in (10, 20, 100)} for p in combinations(ten, 2)}
    sp = {p: [] for p in combinations(ten, 2)}

    for c in giu:
        ds = [hop_nhat([k1.tim(m, k=a.be, be=be)
                        for m in R.tach_truy_van(c.cau_hoi)]),
              k3.tim(c.cau_hoi, k=a.be, be=be),
              k5.tim(c.cau_hoi, k=a.be, be=be)]
        # hạng, không phải điểm — điểm giữa các kênh không so được (schema.py)
        hang = [{u.row_id: -i for i, u in enumerate(d)} for d in ds]
        for (i, j) in combinations(range(3), 2):
            p = (ten[i], ten[j])
            for k in (10, 20, 100):
                x = {u.row_id for u in ds[i][:k]}
                y = {u.row_id for u in ds[j][:k]}
                if x and y:
                    chong[p][k].append(len(x & y) / min(len(x), len(y)))
            s = spearman(hang[i], hang[j])
            if s is not None:
                sp[p].append(s)

    print(f"{'cặp kênh':<24}{'chồng@10':>10}{'chồng@20':>10}"
          f"{'chồng@100':>11}{'Spearman':>11}{'n':>5}")
    print("-" * 71)
    for p in combinations(ten, 2):
        r = [np.mean(chong[p][k]) * 100 if chong[p][k] else float("nan")
             for k in (10, 20, 100)]
        s = np.mean(sp[p]) if sp[p] else float("nan")
        print(f"{p[0] + ' × ' + p[1]:<24}{r[0]:>9.1f}%{r[1]:>9.1f}%"
              f"{r[2]:>10.1f}%{s:>11.3f}{len(sp[p]):>5}")

    print("\nĐỌC BẢNG")
    print("  chồng@20 cao + Spearman cao -> hai kênh THỪA nhau")
    print("  chồng@20 ~0                 -> RRF không có gì cộng hưởng (A14)")
    print("  ở giữa                      -> kênh bổ sung thật")


if __name__ == "__main__":
    main()
