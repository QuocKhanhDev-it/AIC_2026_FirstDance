"""
99_do_menh_de_kenh3.py — Kênh 3 nhận truy vấn theo cách nào? Ba nơi, ba câu trả lời.

    .venv\\Scripts\\python.exe scripts\\99_do_menh_de_kenh3.py

PHÁT HIỆN DẪN TỚI PHÉP ĐO NÀY — MỘT LỆCH GIỮA THƯỚC VÀ THỨ ĐƯỢC ĐO

Cùng một kênh 3, ba đường chạy đưa truy vấn vào ba kiểu khác nhau:

    src/run.py            k3.tim(tach_truy_van(nd))   -> MAX điểm BM25 qua mệnh đề
    scripts/57_,77_,86_   k3.tim(c.cau_hoi)           -> CẢ CÂU, một tài liệu truy vấn
    web/server.py         hop_nhat([k3.tim(m) …])     -> RRF HẠNG qua mệnh đề

Nghĩa là **trọng số 0,5 chốt ở A52 được đo trên cấu hình `run.py` không chạy**,
và A58 ("kênh 3 cần cả câu") cũng vậy. Giao diện soát tay thì lại vẽ ra một bể
ứng viên thứ ba. Đúng loại lệch A23 đã cắn: *"điểm ngoài mà không truy nguyên
được về một lệnh thì nó không dạy ta điều gì cả"*.

Ba cách KHÔNG tương đương, và với BM25 thì khác nhau rõ hơn với cosine:

* **cả câu** — mọi token vào chung một truy vấn, nên tài liệu khớp vài token ở
  mệnh đề này và vài token ở mệnh đề kia được CỘNG DỒN. Truy vấn dài cũng làm
  nhiều token hiếm cùng có mặt.
* **max qua mệnh đề** — tài liệu chỉ ăn điểm của mệnh đề khớp NHẤT; khớp rải
  đều hai mệnh đề không được cộng.
* **RRF hạng qua mệnh đề** — mỗi mệnh đề một phiếu ngang nhau, đúng lập luận
  A51 đã thắng ở kênh 1 (cosine hai mệnh đề khác nhau không so được với nhau).
  ⚠️ Nhưng lập luận đó là về COSINE. Điểm BM25 của hai mệnh đề **cùng thang**
  (cùng công thức, cùng kho), nên nó KHÔNG tự động chuyển sang được — phải đo.

MỐC NỀN LÀ `run.py` NHƯ NÓ ĐANG CHẠY, không phải cấu hình của các script cũ:
kỷ luật "mốc nền phải là cấu hình MẠNH NHẤT hiện có" chỉ có nghĩa khi mốc là
thứ thật sự đang được nộp.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--trong-so", type=float, default=W3)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = [c for c in tap_dev.doc(a.file) if c.loai in ("KIS", "QA")]
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    nhieu = sum(1 for c in giu if len(R.tach_truy_van(c.cau_hoi)) > 1)
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu KIS/QA | "
          f"{nhieu} câu bị tách >1 mệnh đề "
          f"({nhieu / max(len(giu), 1) * 100:.0f}%)\n")

    # Kênh 1 giữ NGUYÊN ở mọi cấu hình — chỉ đổi MỘT thứ mỗi lần (kênh 3).
    _a1 = {}

    def anh(c):
        if c.id not in _a1:
            md = R.tach_truy_van(c.cau_hoi)
            ds = [k1.tim(m, k=a.be) for m in md]
            _a1[c.id] = ds[0] if len(ds) == 1 else hop_nhat(ds)[:a.be]
        return _a1[c.id]

    _k3 = {}

    def ba(c, cach):
        khoa = (c.id, cach)
        if khoa not in _k3:
            md = R.tach_truy_van(c.cau_hoi)
            if cach == "max":                 # đúng run.py hôm nay
                _k3[khoa] = k3.tim(md, k=a.be)
            elif cach == "cau":               # đúng các script đo cũ
                _k3[khoa] = k3.tim(c.cau_hoi, k=a.be)
            else:                             # đúng web/server.py
                ds = [k3.tim(m, k=a.be) for m in md]
                _k3[khoa] = ds[0] if len(ds) == 1 else hop_nhat(ds)[:a.be]
        return _k3[khoa]

    def _nho(f):
        n = {}

        def g(c):
            if c.id not in n:
                n[c.id] = f(c)[:100]
            return n[c.id]
        return g

    def hop(c, cach):
        return hop_nhat([anh(c), ba(c, cach)], trong_so=[1.0, a.trong_so])

    cau_hinh = {
        "0. MỐC — run.py (max mệnh đề)": _nho(lambda c: hop(c, "max")),
        "A. cả câu (script đo cũ)": _nho(lambda c: hop(c, "cau")),
        "B. RRF hạng (web/server)": _nho(lambda c: hop(c, "rrf")),
        "C. kênh 1 một mình": _nho(anh),
    }
    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
