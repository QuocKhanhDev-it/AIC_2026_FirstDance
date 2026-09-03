"""
95_do_khuech_tan_thoi_gian.py — Cho điểm kênh 3 LAN TOẢ theo thời gian.

    python scripts/95_do_khuech_tan_thoi_gian.py

CƠ CHẾ, VÀ VÌ SAO NÓ KHÁC MỌI THỨ ĐÃ BỊ BÁC

A71 đo được chồng@20 giữa kênh 1 và kênh 3 chỉ **3,4%**. Cách đọc thông thường
là "kênh 3 yếu". Cách đọc khác, và nó khớp với bản chất video: **chữ và hình
không xuất hiện cùng một mili-giây**.

    người nói nhắc chủ đề        giây 10   (ASR)
    hình minh hoạ hiện ra        giây 14   (SigLIP2)
    biển hiệu lướt qua           giây  8   (OCR)

RRF cộng theo `row_id`, nên ba sự kiện đó là ba khung khác nhau -> chồng = 0,
và hợp nhất chỉ còn là đan xen (A14).

Khuếch tán: trước khi hợp nhất, cho điểm BM25 của khung `k` lan sang các
keyframe `t` CÙNG VIDEO theo hàm Gauss thời gian

    S'(t) = Σ_k  S(k) · exp( −(t_k − t)² / (2τ²) )

**Giữ nguyên vector ảnh sắc nét**, chỉ làm mềm trường điểm của kênh 3.

⚠️ KHÁC A57 VÀ A70 — hai thứ cùng họ đã bị bác, phải nói rõ khác chỗ nào:

  * A57 làm mượt **vector kênh 1** -> san phẳng phần dư dùng để phân biệt khung
    đáp án với hàng xóm. Ở đây kênh 1 KHÔNG bị đụng.
  * A70 gộp vector theo **đoạn ASR** (trung bình 39 giây) -> mọi khung trong
    đoạn nhận CÙNG một biểu diễn, mất khả năng xếp hạng bên trong đoạn. Ở đây
    τ = 2–6 giây, nhỏ hơn một bậc, và điểm vẫn giảm dần theo khoảng cách nên
    khung gần tâm vẫn hơn khung xa.

τ nằm trong cửa sổ tối thiểu 4 giây của BTC (A9), tức hai khung cách nhau dưới
τ thì BTC coi là cùng một đáp án — khuếch tán trong phạm vi đó không tạo ra
ứng viên "sai" theo luật chấm.

ĐO CẢ HAI THỨ: điểm cuối, và **chồng@20 có thật sự tăng không**. Nếu điểm tăng
mà chồng không tăng thì cơ chế giả thuyết SAI dù kết quả đúng — và lúc đó
không được ghi cơ chế đó vào tài liệu.
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
from rrf import hop_nhat                              # noqa: E402
from schema import Candidate                          # noqa: E402

W3 = 0.5
CAC_TAU = (2.0, 4.0, 6.0)


class KhungTheoVideo:
    """Tra nhanh: các keyframe cùng video nằm trong ±cửa sổ giây."""

    def __init__(self, master):
        self.pts = master.pts_time.values
        self.vid = master.video_id.values
        self.fx = master.frame_idx.values
        thu_tu = np.lexsort((self.pts, self.vid))
        self.sx = thu_tu
        v = self.vid[thu_tu]
        self.dau = {}
        i = 0
        for j in range(1, len(v) + 1):
            if j == len(v) or v[j] != v[i]:
                self.dau[v[i]] = (i, j)
                i = j

    def lan_can(self, row_id: int, cua_so: float):
        """`[(row_id, Δgiây)]` của các keyframe cùng video trong ±cửa sổ."""
        v = self.vid[row_id]
        i, j = self.dau[v]
        t0 = self.pts[row_id]
        doan = self.sx[i:j]
        t = self.pts[doan]
        m = np.abs(t - t0) <= cua_so
        return [(int(r), float(dt)) for r, dt in zip(doan[m], t[m] - t0)]


def khuech_tan(uv: list, kv: KhungTheoVideo, tau: float,
               gioi_han: int = 1000) -> list:
    """Danh sách ứng viên -> danh sách MỚI với điểm đã lan toả theo thời gian."""
    if tau <= 0:
        return uv
    cua_so = 3.0 * tau                      # ngoài 3σ thì hệ số < 0,012
    diem, goc = {}, {}
    hai_tau2 = 2.0 * tau * tau
    for c in uv:
        for r, dt in kv.lan_can(c.row_id, cua_so):
            diem[r] = diem.get(r, 0.0) + c.score * math.exp(-dt * dt / hai_tau2)
            goc.setdefault(r, c)
    ra = []
    for r, d in sorted(diem.items(), key=lambda x: -x[1])[:gioi_han]:
        ra.append(Candidate(row_id=r, video_id=str(kv.vid[r]),
                            frame_idx=int(kv.fx[r]), score=d,
                            source="ocr_asr_kt"))
    return ra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--tau", type=float, nargs="*", default=list(CAC_TAU))
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    kv = KhungTheoVideo(master)
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)} câu | kênh 3 trọng số {W3:g}\n")

    def nho(f):
        c_ = {}

        def g(c):
            if c.id not in c_:
                c_[c.id] = f(c)
            return c_[c.id]
        return g

    anh = nho(lambda c: hop_nhat(
        [k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)]))
    ocr = nho(lambda c: k3.tim(c.cau_hoi, k=a.be))
    kt = {t: nho((lambda t: lambda c: khuech_tan(ocr(c), kv, t, a.be))(t))
          for t in a.tau}

    # ── chồng@20 có tăng không? Cơ chế giả thuyết đứng hay đổ ở đây ──
    print("CHỒNG@20 giữa kênh 1 và kênh 3")
    for ten, f in [("gốc (τ = 0)", ocr)] + [(f"τ = {t:g}s", kt[t])
                                            for t in a.tau]:
        c20 = []
        for c in giu:
            x = {u.row_id for u in anh(c)[:20]}
            y = {u.row_id for u in f(c)[:20]}
            if x and y:
                c20.append(len(x & y) / min(len(x), len(y)))
        print(f"  {ten:<14}{np.mean(c20) * 100:>6.1f}%")
    print()

    cau_hinh = {
        "MỐC: kênh 3 gốc": nho(
            lambda c: hop_nhat([anh(c), ocr(c)], trong_so=[1.0, W3])[:100]),
        "chỉ kênh 1 (chẩn đoán)": nho(lambda c: anh(c)[:100]),
    }
    for t in a.tau:
        cau_hinh[f"khuếch tán τ={t:g}s"] = nho(
            (lambda t: lambda c: hop_nhat([anh(c), kt[t](c)],
                                          trong_so=[1.0, W3])[:100])(t))

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
