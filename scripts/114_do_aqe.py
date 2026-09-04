"""
114_do_aqe.py — α-Query Expansion: mở rộng truy vấn bằng chính top-k trả về.

    python scripts/114_do_aqe.py --file dev/tap_de_that.jsonl

    q' = chuẩn_hoá( q + Σ_{i=1..k} cos(q, d_i)^α · v_{d_i} )

Kỹ thuật chuẩn trong truy hồi ảnh (α-QE, Radenović 2018). Ý: k ảnh đầu tiên
thường đã đúng chủ đề, cộng vector của chúng vào truy vấn thì truy vấn "di
chuyển" về đúng vùng của kho. Mũ α làm ảnh hạng cao đóng góp nhiều hơn.

ÁP Ở TẦNG NÀO — CHỐT TRƯỚC KHI CODE

Áp **trên TỪNG MỆNH ĐỀ, trước RRF**. Lý do: A51 chốt hợp nhất mệnh đề bằng RRF
**hạng**, tức sau bước đó điểm gốc không còn. Mở rộng trên câu đã gộp thì phá
đúng cái logic "mỗi mệnh đề một tiếng nói ngang nhau" đang chạy.

⚠️ RỦI RO THẬT, VÀ NÓ LỚN Ở ĐÂY HƠN Ở BÀI BÁO GỐC

α-QE là **phản hồi giả định** (pseudo-relevance feedback): nó tin rằng top-k
đúng. A92 đo được `R@1 = 0,24` cho KIS — tức ứng viên hạng 1 **sai ở 76% số
câu**. Cộng vector của những ứng viên sai vào truy vấn là kéo truy vấn đi xa
hơn khỏi đích (*query drift*), chứ không phải sửa nó.

Nên bảng dò cả `k` nhỏ (2, 3) lẫn α lớn (3), vì α lớn = tin hạng 1 nhiều hơn,
và với R@1 = 0,24 thì "tin hạng 1" có thể là điều tệ nhất làm được.

⚠️ DỰ ĐOÁN GHI TRƯỚC: k nhỏ + α lớn sẽ TỆ hơn k nhỏ + α nhỏ, vì α lớn khuếch
đại đúng cái ứng viên hạng 1 mà ta biết là sai 76% số lần. Nếu bảng đi ngược
dự đoán này thì giả thuyết "query drift" sai và phải nghĩ lại.
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
CAC_NUT = ((2, 1.0), (2, 3.0), (3, 1.0), (3, 3.0), (5, 3.0))


def tim_vector(k1, q: np.ndarray, k: int) -> list:
    """Như `KenhAnh.tim` nhưng nhận thẳng VECTOR thay vì câu.

    Không gọi lại `tim()` được vì `tim()` tự encode từ chuỗi; ở đây vector đã
    bị sửa. Phần còn lại chép đúng `tim()` để hai đường không lệch nhau.
    """
    sim = k1._nhan(q)
    lay = min(len(sim), k + 200)
    top = np.argpartition(-sim, lay - 1)[:lay]
    top = top[np.argsort(-sim[top])][:k]
    return [Candidate(row_id=int(i), video_id=r.video_id,
                      frame_idx=int(r.frame_idx), score=float(sim[i]),
                      source="clip",
                      meta={"pts_time": float(r.pts_time), "fps": float(r.fps),
                            "kf_n": int(r.kf_n), "title": r.title})
            for i, r in zip(top, k1.master.iloc[top].itertuples())]


def mo_rong(k1, cau: str, k_mo: int, alpha: float, k: int) -> list:
    """Một mệnh đề -> danh sách ứng viên sau khi mở rộng truy vấn."""
    q = k1.encode_text(cau)
    dau = tim_vector(k1, q, k_mo)
    if not dau:
        return []
    v = q.copy()
    for c in dau:
        w = max(c.score, 0.0) ** alpha
        v = v + w * np.asarray(k1.mat[c.row_id], np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    return tim_vector(k1, v, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl"])
    ap.add_argument("--be", type=int, default=100)
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
    print(f"\n{len(giu)} câu | kênh 3 w={W3:g}\n")

    nho, nho3 = {}, {}

    def van(c):
        if c.id not in nho3:
            nho3[c.id] = k3.tim(c.cau_hoi, k=a.be)
        return nho3[c.id]

    def md(m, nut):
        if (m, nut) not in nho:
            nho[(m, nut)] = (k1.tim(m, k=a.be) if nut is None
                             else mo_rong(k1, m, nut[0], nut[1], a.be))
        return nho[(m, nut)]

    def dung(nut):
        n = {}

        def g(c):
            if c.id not in n:
                ds = [md(m, nut) for m in R.tach_truy_van(c.cau_hoi)]
                anh = hop_nhat(ds) if len(ds) > 1 else ds[0]
                n[c.id] = hop_nhat([anh, van(c)], trong_so=[1.0, W3])[:100]
            return n[c.id]
        return g

    cau_hinh = {"1. MỐC: không mở rộng": dung(None)}
    for k_mo, al in CAC_NUT:
        cau_hinh[f"2. α-QE k={k_mo}, α={al:g}"] = dung((k_mo, al))

    print(bao_cao_do_nhay(giu, cau_hinh, master))
    print("\nĐỌC BẢNG: so cặp (k=2, α=1) với (k=2, α=3) và (k=3, α=1) với\n"
          "(k=3, α=3) — đó là phép thử cho giả thuyết query drift, vì α lớn\n"
          "nghĩa là tin hạng 1 nhiều hơn, mà hạng 1 sai ở 76% số câu.")


if __name__ == "__main__":
    main()
