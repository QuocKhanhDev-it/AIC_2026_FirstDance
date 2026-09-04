"""
107_phan_ra_diem.py — Điểm đang mất Ở ĐÂU? Phân rã theo loại câu và độ dài truy vấn.

    python scripts/107_phan_ra_diem.py --file dev/tap_de_that.jsonl

VÌ SAO CẦN BẢNG NÀY

Repo có 91 mục đo, gần như tất cả trả lời "cấu hình A có hơn cấu hình B không".
Không mục nào trả lời **"trong 0,48 điểm đang mất, phần nào nằm ở đâu"** — mà
đó mới là thứ quyết định nên đầu tư vào chỗ nào tiếp theo.

`94_soi_cau_that_bai.py` cho biết 6/49 câu có đáp án ngoài top-1000 và **cả sáu
đều dài >40 từ**. Con số đó tự nó KHÔNG nói gì cho tới khi biết **tỷ lệ nền**:
nếu 45/49 câu vốn đã dài >40 từ thì "6/6 câu hỏng đều dài" là điều hiển nhiên,
không phải phát hiện. Script này in tỷ lệ nền cạnh tỷ lệ hỏng, ở mọi ô.

CÁCH ĐỌC

* `R@k` ở đây là tỷ lệ câu có đáp án trong top-k, tính RIÊNG từng ô.
* Cột `mất` = phần điểm mà ô đó đang bỏ lại trên bàn, đã NHÂN với số câu trong
  ô — tức "chữa xong ô này thì tổng điểm tăng tối đa bao nhiêu".
  Ô ít câu mà điểm thấp thì `mất` vẫn nhỏ; đừng đuổi theo tỷ lệ phần trăm.
* `hạng trung vị` chỉ tính trên các câu TÌM ĐƯỢC; câu trượt ghi riêng ở `∞`.

⚠️ TRAKE chấm ở tầng KÊNH nên điểm của nó CAO HƠN điểm nộp thật (A63: 37% ở
±2s). Bảng tách riêng loại TRAKE để con số đó không trộn vào KIS/QA.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import MOC                             # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
DUNG_SAI = 2.0


def hang_dung(uv, c, master, dung_sai):
    """Hạng (1-based) của ứng viên ĐÚNG đầu tiên, hoặc None nếu không có."""
    pts, vid = master.pts_time.values, master.video_id.values
    dung = c.row_id_dung
    if c.loai == "TRAKE":
        dung = [r for b in dung for r in b]
    moc_t = [(str(vid[r]), float(pts[r])) for r in dung]
    for i, u in enumerate(uv, 1):
        t = float(pts[u.row_id])
        if any(v == u.video_id and abs(t - t0) <= dung_sai for v, t0 in moc_t):
            return i
    return None


def diem(hangs):
    """Final Score của một ô = trung bình R@{1,5,20,50,100}."""
    if not hangs:
        return 0.0
    return float(np.mean([np.mean([h is not None and h <= m for h in hangs])
                          for m in MOC]))


def bang(ten_o, o, tong_cau):
    print(f"\n{ten_o}")
    print(f"  {'ô':<22}{'câu':>5}{'điểm':>9}{'mất':>9}  "
          + "".join(f"R@{m:<5}" for m in MOC) + "  hạng trung vị  trượt")
    print("  " + "-" * 96)
    for ten, hangs in o:
        d = diem(hangs)
        mat = (1.0 - d) * len(hangs) / tong_cau
        tim = [h for h in hangs if h is not None]
        tv = f"{int(np.median(tim))}" if tim else "—"
        r = "".join(f"{np.mean([h is not None and h <= m for h in hangs]):<7.2f}"
                    for m in MOC)
        print(f"  {ten:<22}{len(hangs):>5}{d:>9.4f}{mat:>9.4f}  {r}  "
              f"{tv:>12}  {sum(h is None for h in hangs):>5}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--dung-sai", type=float, default=DUNG_SAI)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]

    hang, dai, loai = {}, {}, {}
    for c in giu:
        me = R.tach_truy_van(c.cau_hoi)
        uv = hop_nhat([hop_nhat([k1.tim(m, k=a.be) for m in me]),
                       k3.tim(c.cau_hoi, k=a.be)], trong_so=[1.0, W3])[:100]
        hang[c.id] = hang_dung(uv, c, master, a.dung_sai)
        dai[c.id] = len(c.cau_hoi.split())
        loai[c.id] = c.loai

    n = len(giu)
    tong = diem([hang[c.id] for c in giu])
    print(f"\n{n} câu | dung sai ±{a.dung_sai:g}s | kênh 3 w={W3:g}")
    print(f"ĐIỂM TỔNG: {tong:.4f}   -> đang mất {1 - tong:.4f}\n")

    o = defaultdict(list)
    for c in giu:
        o[loai[c.id]].append(hang[c.id])
    bang("THEO LOẠI CÂU", sorted(o.items()), n)

    def nhom_dai(w):
        return "≤40 từ" if w <= 40 else ("41-70 từ" if w <= 70 else ">70 từ")

    o = defaultdict(list)
    for c in giu:
        o[nhom_dai(dai[c.id])].append(hang[c.id])
    thu_tu = ["≤40 từ", "41-70 từ", ">70 từ"]
    bang("THEO ĐỘ DÀI TRUY VẤN (tỷ lệ nền nằm ở cột `câu`)",
         [(k, o[k]) for k in thu_tu if k in o], n)

    o = defaultdict(list)
    for c in giu:
        o[f"{loai[c.id]} · {nhom_dai(dai[c.id])}"].append(hang[c.id])
    bang("CHÉO LOẠI × ĐỘ DÀI", sorted(o.items()), n)

    print("\nĐỌC BẢNG: cột `mất` đã nhân với số câu trong ô, nên nó xếp hạng "
          "ĐÚNG\nthứ tự nên đầu tư. Ô có điểm thấp nhất chưa chắc là ô đáng "
          "chữa nhất.")


if __name__ == "__main__":
    main()
