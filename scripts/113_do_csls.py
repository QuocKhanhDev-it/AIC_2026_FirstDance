"""
113_do_csls.py — CSLS: phạt keyframe "gần mọi thứ" có kéo đáp án lên hạng 1 không?

    python scripts/113_do_csls.py --file dev/tap_de_that.jsonl

    s(q, d) = 2·cos(q, d) − λ·r_K(d)          λ dò lưới, xem 112_

VÌ SAO NHẮM VÀO ĐÂY

A92 phân rã 0,4769 điểm đang mất: **KIS mất 0,2962**, và trong đó `R@1 = 0,24`
so với `R@100 = 0,81`. Đáp án nằm trong bể ở 81% số câu nhưng chỉ đứng đầu ở
24%. Bài toán là **xếp hạng trong top-100**, và hub là một nghi phạm cụ thể:
nếu vài keyframe "gần mọi thứ" luôn chen lên trên, chúng đẩy đáp án xuống ở
đúng những mốc R@1 và R@5 đắt nhất.

⚠️ ĐO Ở TẦNG NÀO. CSLS sửa **điểm cosine của kênh 1**, tức nó phải áp TRƯỚC
khi hợp nhất mệnh đề. Sau khi RRF hạng đã chạy thì điểm gốc không còn tồn tại,
chỉ còn thứ hạng — áp CSLS ở đó là vô nghĩa. Nên bảng có cả dòng "chỉ kênh 1"
để thấy CSLS làm gì khi chưa bị RRF và kênh 3 pha loãng.

⚠️ DỰ ĐOÁN GHI TRƯỚC: tôi cho rằng hiệu sẽ nhỏ và nhiều khả năng ⚪ hoặc 🟡.
Lý do: `112_` in độ lệch chuẩn của `r_K`; nếu nó nhỏ so với khoảng biến thiên
cosine (0,25-0,40 theo `rrf.py`) thì phép trừ gần như là một hằng số cộng vào
mọi ứng viên, và **hằng số không đổi được thứ hạng nào**. Bảng dưới sẽ nói rõ.
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
from schema import Candidate                          # noqa: E402

W3 = 0.5
CAC_LAMBDA = (0.1, 0.25, 0.5, 1.0)


def ap_csls(uv: list, hub: np.ndarray, lam: float) -> list:
    """Xếp lại một danh sách ứng viên theo `2·cos − λ·r_K`.

    ⚠️ `Candidate.score` của kênh 1 LÀ cosine (xem `dense.KenhAnh.tim`), nên
    phép này đúng nghĩa. Nếu ai đó đổi kênh 1 sang trả về điểm khác thì hàm
    này lặng lẽ tính sai — đó là lý do nó chỉ được gọi trên đầu ra kênh 1,
    không bao giờ trên đầu ra RRF.
    """
    if lam == 0.0:
        return uv
    moi = []
    for c in uv:
        d = 2.0 * c.score - lam * float(hub[c.row_id])
        moi.append(Candidate(row_id=c.row_id, video_id=c.video_id,
                             frame_idx=c.frame_idx, score=d,
                             source=c.source, meta=c.meta))
    moi.sort(key=lambda x: -x.score)
    return moi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl"])
    ap.add_argument("--hub", default=None, type=Path)
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--lam", type=float, nargs="*", default=list(CAC_LAMBDA))
    a = ap.parse_args()

    f_hub = a.hub or (a.index / "hubness_clip_gopt.npy")
    if not f_hub.exists():
        raise SystemExit(f"❌ chưa có {f_hub}\n   chạy: python "
                         f"scripts/112_tinh_hubness.py")
    hub = np.load(f_hub)

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

    print(f"\n{len(giu)} câu | kênh 3 w={W3:g}")
    print(f"r_K: trung vị {np.median(hub):.4f}, độ lệch chuẩn {hub.std():.4f}")
    print(f"     -> λ=1 dịch điểm đi tối đa {hub.max() - hub.min():.4f}, "
          f"trong khi cosine trải ~0,15 (0,25-0,40).")
    print(f"     Nếu độ lệch chuẩn ≪ 0,15 thì CSLS gần như là hằng số cộng "
          f"và KHÔNG đổi được thứ hạng.\n")

    nho1, nho3 = {}, {}

    def menh_de(m):
        if m not in nho1:
            nho1[m] = k1.tim(m, k=a.be)
        return nho1[m]

    def van(c):
        if c.id not in nho3:
            nho3[c.id] = k3.tim(c.cau_hoi, k=a.be)
        return nho3[c.id]

    def dung(lam, chi_kenh1=False):
        n = {}

        def g(c):
            if c.id not in n:
                ds = [ap_csls(menh_de(m), hub, lam)
                      for m in R.tach_truy_van(c.cau_hoi)]
                anh = hop_nhat(ds) if len(ds) > 1 else ds[0]
                n[c.id] = (anh[:100] if chi_kenh1 else
                           hop_nhat([anh, van(c)], trong_so=[1.0, W3])[:100])
            return n[c.id]
        return g

    cau_hinh = {"1. MỐC: không CSLS (λ=0)": dung(0.0)}
    for lam in a.lam:
        cau_hinh[f"2. CSLS λ={lam:g}"] = dung(lam)
    cau_hinh["3. chẩn đoán: chỉ kênh 1, không CSLS"] = dung(0.0, True)
    best = max(a.lam)
    cau_hinh[f"3. chẩn đoán: chỉ kênh 1, λ={best:g}"] = dung(best, True)

    print(bao_cao_do_nhay(giu, cau_hinh, master))
    print("\nĐỌC BẢNG: hai dòng chẩn đoán so VỚI NHAU cho biết CSLS làm gì khi\n"
          "chưa bị RRF và kênh 3 pha loãng. Nếu ở đó cũng ⚪ thì cơ chế sai,\n"
          "không phải bị pha loãng.")


if __name__ == "__main__":
    main()
