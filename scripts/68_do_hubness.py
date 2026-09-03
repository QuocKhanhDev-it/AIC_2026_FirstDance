"""
68_do_hubness.py — Chữa "hubness": vài khung hình lọt top của MỌI truy vấn.

    python scripts/68_do_hubness.py --chuan-bi     # tính thống kê (một lần)
    python scripts/68_do_hubness.py                # đo

TRIỆU CHỨNG ĐÃ ĐO ĐƯỢC, KHÔNG PHẢI GIẢ THUYẾT

A54: **R@20 = 0,6122 nhưng R@1 = 0,2041.** Với 61% số câu, đáp án đã nằm trong
20 dòng đầu — chỉ là không ở dòng đầu. A55 thử ba cách xếp lại bằng tín hiệu
sẵn có và lấy được ~0. Nhưng cả ba đều nhìn vào **quan hệ giữa các ứng viên của
CÙNG một truy vấn**. Hướng này nhìn thứ khác hẳn.

HUBNESS LÀ GÌ

Trong không gian nhiều chiều, một số điểm trở thành **"hub"** — chúng gần với
*mọi* truy vấn, không riêng truy vấn nào. Khung hình chung chung (người nói
trước micro, cảnh đường phố, phông studio) có vector nằm gần tâm đám mây, nên
cosine với truy vấn nào cũng cao vừa phải. Chúng chiếm mất hạng 1–5, đẩy khung
đặc trưng — thứ chỉ khớp ĐÚNG một truy vấn — xuống hạng 10–20.

Đó chính xác là hình dạng của R@1 thấp mà R@20 cao.

HAI CÁCH CHỮA, ĐỀU KHÔNG CẦN MODEL

**1. Trừ tâm (centering).** Bớt vector trung bình của kho khỏi mọi vector ảnh
rồi chuẩn hoá lại. Thành phần "chung của mọi ảnh" bị bỏ đi, chỉ còn phần riêng.

**2. QB-Norm / inverted softmax.** Dùng chính **bể truy vấn đã mã hoá** (1.239
chuỗi) làm mẫu: khung nào được cả bể chấm cao thì bị phạt.

    điểm' = sim(q,i)/τ − log Σ_{q' ∈ bể} exp(sim(q',i)/τ)

Mẫu số không phụ thuộc truy vấn đang hỏi — tính SẴN một lần, lúc chạy chỉ là
một phép trừ trên mảng 177.321 phần tử.

⚠️ **Tự loại mình khỏi bể.** Bể chứa cả chính truy vấn đang đo. Để nguyên là
truy vấn tự phạt mình, và đó là một dạng rò tinh vi. Ở đây trừ đúng số hạng của
nó ra khỏi tổng — làm được vì tổng là tổng các `exp`, và `sim` của chính nó thì
đã tính rồi.

⚠️ Không nhóm nào trong 9 repo đối chiếu làm việc này. Nghĩa là chưa ai xác
nhận nó có ích cho bài toán này — nên phải đo, đừng tin.
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
TAU = (0.01, 0.02, 0.05)
LO = 20_000
F_THONG_KE = GOC / "index" / "hubness_gopt.npz"


def chuan_bi(k1: KenhAnhCache, ra: Path):
    """Một lượt quét kho: tâm ảnh + tổng exp theo từng τ."""
    Q = np.stack([k1._cache[c] for c in k1._cache]).astype(np.float32)
    n, d = k1.mat.shape
    print(f"bể truy vấn: {Q.shape[0]:,} chuỗi | kho: {n:,} ảnh × {d}")

    tong = np.zeros(d, dtype=np.float64)
    S = {t: np.zeros(n, dtype=np.float64) for t in TAU}
    for i in range(0, n, LO):
        M = np.asarray(k1.mat[i:i + LO], dtype=np.float32)
        tong += M.sum(0)
        sim = M @ Q.T                                  # (lô, số truy vấn)
        for t in TAU:
            S[t][i:i + LO] = np.exp(sim / t).sum(1)
        print(f"  {min(i + LO, n):,}/{n:,}", end="\r", flush=True)

    mu = (tong / n).astype(np.float32)
    # Chuẩn của ảnh SAU khi trừ tâm — cần để chuẩn hoá lại mà không giữ cả
    # ma trận đã trừ (545 MB nữa).
    chuan = np.empty(n, dtype=np.float32)
    for i in range(0, n, LO):
        M = np.asarray(k1.mat[i:i + LO], dtype=np.float32) - mu
        chuan[i:i + LO] = np.linalg.norm(M, axis=1)

    np.savez(ra, mu=mu, chuan=chuan, tau=np.array(TAU),
             **{f"S{j}": S[t] for j, t in enumerate(TAU)})
    print(f"\n✅ {ra}")
    for t in TAU:
        lg = np.log(S[t])
        print(f"   τ={t:<5g} log-tổng: min {lg.min():.2f} trung vị "
              f"{np.median(lg):.2f} max {lg.max():.2f}  "
              f"(chênh {lg.max() - lg.min():.2f} — càng lớn càng nhiều hub)")


class KenhTru(KenhAnhCache):
    """Kênh 1 có trừ tâm và/hoặc phạt hub. `tim()` thừa kế nguyên vẹn."""

    def cai(self, mu=None, chuan=None, phat=None, tau=1.0, mu_q=None):
        self.mu, self.chuan, self.phat, self.tau = mu, chuan, phat, tau
        # Trừ tâm ĐÚNG CÁCH là trừ ở CẢ HAI phía: ảnh bớt tâm ảnh, truy vấn
        # bớt tâm truy vấn. Chỉ trừ một phía là dịch đám mây ảnh đi mà để đám
        # mây truy vấn đứng yên — hai bên lệch nhau, cosine mất nghĩa.
        self.mu_q = mu_q
        return self

    def _nhan(self, q: np.ndarray) -> np.ndarray:
        if self.mu_q is not None:
            q = q - self.mu_q
            q = q / (np.linalg.norm(q) + 1e-9)
        n = self.mat.shape[0]
        ra = np.empty(n, dtype=np.float32)
        for i in range(0, n, self.LO):
            M = np.asarray(self.mat[i:i + self.LO], dtype=np.float32)
            if self.mu is not None:
                M = M - self.mu
                ra[i:i + self.LO] = (M @ q) / (self.chuan[i:i + self.LO] + 1e-9)
            else:
                ra[i:i + self.LO] = M @ q
        if self.phat is None:
            return ra
        # QB-Norm: bớt phần "khung này ai cũng thích", SAU khi tự loại mình.
        con = np.maximum(self.phat - np.exp(ra / self.tau), 1e-30)
        return ra / self.tau - np.log(con).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--chuan-bi", action="store_true")
    ap.add_argument("--nhanh", action="store_true",
                    help="chỉ đo hai biến thể trừ tâm, bỏ QB-Norm")
    a = ap.parse_args()

    cache = str(a.index / "truy_van_gopt.npz")
    if a.chuan_bi:
        chuan_bi(KenhAnhCache(str(a.index), cache, matrix="clip_gopt.npy"),
                 F_THONG_KE)
        return
    if not F_THONG_KE.exists():
        raise SystemExit(f"Chưa có {F_THONG_KE}. Chạy `--chuan-bi` trước "
                         f"(một lượt quét kho, vài phút).")

    z = np.load(F_THONG_KE)
    mu, chuan, taus = z["mu"], z["chuan"], list(z["tau"])
    S = {t: z[f"S{j}"] for j, t in enumerate(taus)}

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    def kenh(**kw):
        return KenhTru(str(a.index), cache, matrix="clip_gopt.npy").cai(**kw)

    goc = kenh()
    giu = [c for c in cau if not goc.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu | bể {a.be}\n")

    ocr = {}

    def lam(k1):
        nho = {}

        def g(c):
            if c.id not in nho:
                anh = hop_nhat([k1.tim(m, k=a.be)
                                for m in R.tach_truy_van(c.cau_hoi)])
                if c.id not in ocr:
                    ocr[c.id] = k3.tim(c.cau_hoi, k=a.be)
                nho[c.id] = hop_nhat([anh, ocr[c.id]], trong_so=[1.0, W3])
            return nho[c.id]
        return g

    # Tâm của bể truy vấn — cần cho biến thể trừ tâm CẢ HAI PHÍA.
    mu_q = np.stack([goc._cache[c] for c in goc._cache]).astype(np.float32).mean(0)

    cau_hinh = {
        "1. mốc: run.py": lam(goc),
        "2. trừ tâm CHỈ phía ảnh": lam(kenh(mu=mu, chuan=chuan)),
        "3. trừ tâm CẢ HAI phía": lam(kenh(mu=mu, chuan=chuan, mu_q=mu_q)),
    }
    if a.nhanh:
        print(bao_cao_do_nhay(giu, cau_hinh, master))
        return
    for t in taus:
        cau_hinh[f"4. QB-Norm τ={t:g}"] = lam(kenh(phat=S[t], tau=float(t)))
    t0 = float(taus[1])
    cau_hinh[f"5. QB + trừ tâm (τ={t0:g})"] = lam(
        kenh(mu=mu, chuan=chuan, phat=S[t0], tau=t0))

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
