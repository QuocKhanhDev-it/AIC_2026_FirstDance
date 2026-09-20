"""
85_do_hop_nhat_diem.py — Hợp nhất KÊNH bằng ĐIỂM chuẩn hoá có hơn RRF hạng không?

    python scripts/85_do_hop_nhat_diem.py

VÌ SAO ĐO LẠI MỘT THỨ `rrf.py` ĐÃ TỪ CHỐI TỪ ĐẦU

`rrf.py` không cộng điểm vì thang điểm các kênh không so được. Đúng, nhưng đó
là lý do NÉ, chưa từng là một phép đo. A84 mới cho lý do phải đo thật:

    chồng@20 giữa kênh 1 và kênh 5 = 4,3%   |   Spearman = 0,093
    chồng@20 giữa kênh 1 và kênh 3 = 4,3%   |   Spearman = 0,203

Các kênh gần như KHÔNG BAO GIỜ đề cử cùng `row_id`. RRF chỉ cộng hưởng khi có
trùng `row_id`; không trùng thì nó chỉ **đan xen** — và A14 đo được đan xen làm
TỆ ĐI, vì mỗi ứng viên tốt của kênh mạnh bị một ứng viên của kênh yếu đẩy lùi
một bậc. Cộng điểm không có tính chất đó: ứng viên vượt trội giữ được BIÊN ĐỘ.

CHỈ ĐỔI MỘT THỨ

Mốc nền là cấu hình `run.py` đang chạy (kênh 1 tách mệnh đề + RRF hạng theo
A51, hợp với kênh 3 trọng số 0,5 theo A52). Thứ duy nhất đổi là **cơ chế hợp
nhất ở tầng KÊNH**. Hợp nhất mệnh đề TRONG kênh 1 vẫn là RRF hạng ở mọi dòng —
A51 đã thắng ở đó, không đụng vào.

BỐN NÚT ĐƯỢC DÒ, MỖI NÚT MỘT CÂU HỎI RIÊNG

  * cơ chế chuẩn hoá  — z / min-max / sigmoid có nhiệt độ
  * nhiệt độ `tau`    — thay cho `logit_scale` của SigLIP2, xem docstring
                        `hop_diem.py`: σ đơn điệu nên nó KHÔNG đổi thứ hạng nội
                        bộ kênh, chỉ đổi biên độ khi cộng
  * nắn BM25 `log1p`  — phân phối BM25 lệch nặng, đuôi kéo z-score
  * giá trị bù        — "min" hay 0 cho ứng viên vắng mặt (bẫy A60)

⚠️ CHƯA ĐỔI MẶC ĐỊNH. `run.py` vẫn RRF cho tới khi thắng ở CẢ HAI mức dung sai.
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
from hop_diem import hop_nhat_diem                    # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5                                              # A52


def nho(f):
    """Nhớ theo `id` câu — mỗi cấu hình bị gọi lại ở hai mức dung sai."""
    cache = {}

    def g(c):
        if c.id not in cache:
            cache[c.id] = f(c)[:100]
        return cache[c.id]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--w3", type=float, default=W3)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    if len(giu) < len(cau):
        print(f"⚠️ loại {len(cau) - len(giu)} câu thiếu chuỗi trong cache")
    print(f"{'+'.join(f.stem for f in a.file)}: đo {len(giu)}/{len(cau)} câu | kênh 3 trọng số "
          f"{a.w3:g}\n")

    # Kênh 1 y hệt `run.py` sau A51: mỗi mệnh đề một truy vấn, gộp RRF hạng.
    anh = nho(lambda c: hop_nhat(
        [k1.tim(m, k=100) for m in R.tach_truy_van(c.cau_hoi)]))
    ocr = nho(lambda c: k3.tim(c.cau_hoi, k=100))

    def diem(cach, tau=1.0, nan_bm25=False, bu="min"):
        return nho(lambda c: hop_nhat_diem(
            [anh(c), ocr(c)], trong_so=[1.0, a.w3], cach=cach, tau=tau,
            truoc=[None, "log1p" if nan_bm25 else None], bu=bu))

    cau_hinh = {
        "1. MỐC: RRF hạng": nho(
            lambda c: hop_nhat([anh(c), ocr(c)], trong_so=[1.0, a.w3])),
        "2. điểm z-score": diem("z"),
        "3. điểm min-max": diem("minmax"),
        "4. điểm sigmoid tau=2": diem("sigmoid", tau=2.0),
        "5. điểm sigmoid tau=8": diem("sigmoid", tau=8.0),
        "6. z-score + log1p BM25": diem("z", nan_bm25=True),
        "7. min-max + log1p BM25": diem("minmax", nan_bm25=True),
        "8. z-score, bù 0 (bẫy A60)": diem("z", bu="zero"),
        "9. chỉ kênh 1 (chẩn đoán)": anh,
    }

    print(bao_cao_do_nhay(giu, cau_hinh, master))
    print("\nĐỌC BẢNG: dòng 2-7 khác dòng 1 ĐÚNG MỘT THỨ — cơ chế hợp nhất ở "
          "tầng kênh.\nHợp nhất mệnh đề trong kênh 1 vẫn là RRF hạng ở mọi "
          "dòng (A51).")


if __name__ == "__main__":
    main()
