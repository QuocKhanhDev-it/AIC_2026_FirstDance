"""
74_do_lai_cham_diem_objects.py — Kênh 4 chết vì Ý TƯỞNG hay vì CÔNG THỨC CHẤM?

    python scripts/74_do_lai_cham_diem_objects.py

CHẨN ĐOÁN DẪN TỚI ĐÂY

A61 đo kênh 4 được 0,0125 đứng một mình. Nhưng soi kỹ thì dữ liệu LÀNH (95% độ
phủ, 514 nhãn), bảng ánh xạ Việt–Anh LÀNH (phủ 97,3% số lần xuất hiện), và kênh
RÚT NHÃN ĐÚNG — ví dụ `kis-DE1-20` ("2 người cầm dù") rút ra đúng `Umbrella`,
mà `Umbrella` chỉ có ở 355/168.470 keyframe. Vậy mà đáp án vẫn không lọt top-100.

Lỗi nằm ở `object_score()`:

    điểm = Σ (độ tin cậy × IDF) trên MỌI detection khớp

Hai chỗ hỏng, cả hai đều làm khung "chung chung" thắng khung đặc trưng:

**1. Mỗi DETECTION cộng một lần.** Khung có 8 người thì ăn 8 × IDF(Person).
Khung có đúng 1 cái ô hiếm chỉ ăn 1 × IDF(Umbrella). Đếm số lượng vật thể chứ
không đếm SỰ CÓ MẶT.

**2. Cộng trên mọi nhãn để nhãn phổ biến át nhãn hiếm.** `kis-DE1-20` rút ra
[Building, Clothing, House, Person, Shirt, Umbrella]. Một cảnh phố có đủ 5 nhãn
đầu (đều phổ biến) sẽ vượt khung DUY NHẤT có Umbrella.

BỐN CÁCH SỬA ĐEM ĐO

  1. **gộp detection**: mỗi nhãn chỉ tính MỘT lần (lấy độ tin cậy cao nhất)
  2. **lấy MAX thay vì tổng**: điểm = nhãn khớp mạnh nhất, không cộng dồn
  3. **chỉ nhãn hiếm**: bỏ nhãn IDF < 3 khỏi truy vấn trước khi chấm
  4. **lọc theo nhãn hiếm nhất**: chỉ xét khung CÓ nhãn hiếm nhất mà câu nhắc

⚠️ Đây là ĐO LẠI MỘT KÊNH ĐANG HỎNG — đúng bài học A11. Nếu bản sửa vẫn ~0 thì
kênh 4 chết vì ý tưởng; nếu nó bật lên thì A25 và A61 đã kết luận về một lỗi
cài đặt chứ không phải về objects.
"""

import argparse
import math
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
from objects import KenhObjects                       # noqa: E402
from rrf import hop_nhat                              # noqa: E402
from schema import Candidate                          # noqa: E402

W3 = 0.5
IDF_HIEM = 3.0


class Kenh4Moi:
    """Kênh objects với công thức chấm thay được."""

    def __init__(self, index: Path, master, cach: str):
        self.master, self.cach = master, cach
        self.goc = KenhObjects(str(index), master)      # dùng lại `nhan()`
        o = pd.read_parquet(index / "objects.parquet")
        # GỘP DETECTION: mỗi (khung, nhãn) một dòng, giữ độ tin cậy cao nhất.
        self.o = o.groupby(["row_id", "label"], as_index=False).score.max()
        n = o.row_id.nunique()
        self.idf = (n / o.groupby("label").row_id.nunique()).apply(math.log)

    def tim(self, cau, k: int = 100, be=None):
        nhan = sorted(set(self.goc.nhan(cau) if isinstance(cau, str)
                          else {x for c in cau for x in self.goc.nhan(c)}))
        if self.cach in ("chi_hiem", "loc_hiem_nhat"):
            nhan = [x for x in nhan if self.idf.get(x, 0) >= IDF_HIEM]
        if not nhan:
            return []
        if self.cach == "loc_hiem_nhat":
            nhan = [max(nhan, key=lambda x: self.idf.get(x, 0))]

        sub = self.o[self.o.label.isin(nhan)]
        if sub.empty:
            return []
        w = sub.score.values * sub.label.map(self.idf).fillna(0.0).values
        g = pd.Series(w).groupby(sub.row_id.values)
        diem = (g.max() if self.cach == "max" else g.sum())

        d = np.zeros(len(self.master), dtype=np.float32)
        d[diem.index.values] = diem.values
        if be is not None:
            d = np.where(np.asarray(be, dtype=bool), d, 0.0)
        top = np.argsort(-d)[:k]
        m, ra = self.master, []
        for i in top:
            if d[i] <= 0:
                break
            ra.append(Candidate(row_id=int(i), video_id=m.video_id.iloc[i],
                                frame_idx=int(m.frame_idx.iloc[i]),
                                score=float(d[i]), source="objects",
                                meta={"pts_time": float(m.pts_time.iloc[i])}))
        return ra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    k4_cu = KenhObjects(str(a.index), master)
    moi = {c: Kenh4Moi(a.index, master, c)
           for c in ("gop", "max", "chi_hiem", "loc_hiem_nhat")}

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu\n")

    nen = {}

    def phan(c):
        if c.id not in nen:
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
            nen[c.id] = (anh, k3.tim(c.cau_hoi, k=a.be))
        return nen[c.id]

    def _nho(f):
        n = {}

        def g(c):
            if c.id not in n:
                n[c.id] = f(c)[:100]
            return n[c.id]
        return g

    TEN = {"gop": "gộp detection", "max": "lấy MAX",
           "chi_hiem": "chỉ nhãn IDF≥3", "loc_hiem_nhat": "chỉ nhãn hiếm NHẤT"}
    cau_hinh = {
        "0. mốc: run.py (không kênh 4)":
            _nho(lambda c: hop_nhat(list(phan(c)), trong_so=[1.0, W3])),
        "1. chỉ kênh 4 CŨ (chẩn đoán)": _nho(lambda c: k4_cu.tim(c.cau_hoi, k=a.be)),
    }
    for ten, kk in moi.items():
        cau_hinh[f"2. chỉ kênh 4 — {TEN[ten]}"] = _nho(
            (lambda kk: lambda c: kk.tim(c.cau_hoi, k=a.be))(kk))
    for ten in ("chi_hiem", "loc_hiem_nhat"):
        cau_hinh[f"3. mốc + kênh 4 {TEN[ten]} (0,25)"] = _nho(
            (lambda kk: lambda c: hop_nhat(
                [*phan(c), kk.tim(c.cau_hoi, k=a.be)],
                trong_so=[1.0, W3, 0.25]))(moi[ten]))

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
