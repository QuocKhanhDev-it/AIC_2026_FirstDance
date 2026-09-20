"""
92_do_lai_ghep_trake.py — TRAKE: K-best sâu cho top-5 + lưới an toàn cho phần đuôi.

    python scripts/92_do_lai_ghep_trake.py

CƠ CHẾ, ĐO ĐƯỢC CHỨ KHÔNG SUY

A74/A78 đo K-best trên 20 câu: +0,0900 ở ±2s, ngưỡng nhiễu 0,0937 — thiếu 4%.
Bảng từng câu chỉ ra một câu ăn mất một phần tư hiệu ứng:

    trake-L25-004   CŨ 0,4800  ->  K-best 0,0000   (oracle 0,8000)

Hạng của video ĐÚNG trong 20 câu: 14 câu hạng 1, 4 câu hạng 2–5, và **2 câu
ngoài top-5** — `trake-L25-002` hạng 8, `trake-L25-004` **hạng 23**.

K-best chỉ xét **top-5 video**, nên với hai câu đó nó sinh KHÔNG một giả thuyết
nào -> đúng 0 điểm. Cách CŨ rải 1 dòng cho mỗi trong 100 video nên vẫn chạm
tới hạng 23 và được 0,4800.

    K-best đổi BỀ RỘNG lấy CHIỀU SÂU.
    18/20 câu thì đổi là lời. 2/20 câu thì mất trắng.

GHÉP LẠI: giữ chiều sâu, gắn lại lưới an toàn

Dành `n_sau` dòng cuối cho **1 chuỗi tốt nhất mỗi video** ở hạng 6 trở đi. Mỗi
dòng ở đó rẻ (chỉ lấy đi một biến thể sâu của top-5) nhưng cứu được cả câu khi
video đúng nằm ngoài top-5.

⚠️ Khác hẳn "định mức chỗ cho kênh 3" (A75, thua 0 thắng/4 thua): ở đó phần
được dành CHỖ vốn đã nằm trong top-100 rồi nên định mức không thêm gì. Ở đây
phần được dành chỗ **hiện đang hoàn toàn vắng mặt** — đó là khác biệt quyết
định, và nó đo được ở dòng `trake-L25-004`.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import (bao_cao_tu_bang,               # noqa: E402
                       cham_trake_nhieu_muc)
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

_s = importlib.util.spec_from_file_location(
    "m78", GOC / "scripts" / "78_do_kbest_trake.py")
m78 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m78)

_s9 = importlib.util.spec_from_file_location(
    "m89", GOC / "scripts" / "89_do_chon_video_trake.py")
m89 = importlib.util.module_from_spec(_s9)
_s9.loader.exec_module(m89)

W3 = 0.5
GIAN = 3.0                       # giãn tốt nhất theo A74
TY_LE = (0.4, 0.25, 0.15, 0.12, 0.08)     # ứng viên tốt nhất theo A78


def lap_ghep(cac, master, n_sau, so_dong=100):
    """K-best sâu cho top-5, rồi `n_sau` dòng cho mỗi video hạng 6 trở đi."""
    pts = master.pts_time.values
    diem_v, theo_video = m89.xep_video(cac, m89.nhan)
    if not diem_v:
        return []
    xep = sorted(diem_v, key=lambda v: -diem_v[v])

    def chuoi(v, k):
        uv = theo_video[v]
        for x in uv:
            x.sort(key=lambda t: -t[1])
        return m78.beam_video([x[:20] for x in uv], pts, k)

    n_tren = so_dong - n_sau
    ra = []
    for v, w in zip(xep[:5], TY_LE):
        ra += chuoi(v, max(1, round(n_tren * w)))
        if len(ra) >= n_tren:
            break
    ra = ra[:n_tren]
    # lưới an toàn: 1 chuỗi tốt nhất cho mỗi video hạng 6 trở đi
    for v in xep[5:5 + n_sau]:
        ra += chuoi(v, 1)
    return ra[:so_dong]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl",
        GOC / "dev" / "tap_dev_trake.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--sau", type=int, nargs="*", default=[10, 20, 30, 50])
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "TRAKE"]
    giu = [c for c in cau
           if not any(k1.co_du(R.tach_truy_van(m))
                      for m in R.tach_su_kien(c.cau_hoi))]
    print(f"{len(giu)}/{len(cau)} câu TRAKE đo được\n")

    nen = {}
    for c in giu:
        cac = []
        for sk in R.tach_su_kien(c.cau_hoi):
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(sk)])
            cac.append(hop_nhat([anh, k3.tim(sk, k=a.be)], trong_so=[1.0, W3]))
        nen[c.id] = cac

    m78.CACH_NHAU[0] = GIAN

    def nho(f):
        c_ = {}

        def g(c):
            if c.id not in c_:
                c_[c.id] = f(c)
            return c_[c.id]
        return g

    tra = {}
    for r_, v_, f_ in zip(master.row_id.values, master.video_id.values,
                          master.frame_idx.values):
        tra.setdefault((v_, int(f_)), set()).add(int(r_))

    # MỐC = K-best thuần, tức đúng cấu hình A74 đang đề xuất bật.
    cau_hinh = {
        "K-best thuần ← MỐC": nho(lambda c: lap_ghep(nen[c.id], master, 0)),
        "CŨ 1 dòng/video": nho(lambda c: [
            [tra.get((d.video_id, f), set()) for f in d.frame_idxs]
            for d in R.dung_trake(nen[c.id], master)]),
    }
    for n in a.sau:
        cau_hinh[f"K-best + {n} dòng đuôi"] = nho(
            (lambda n: lambda c: lap_ghep(nen[c.id], master, n))(n))

    bang = {ten: cham_trake_nhieu_muc(giu, f, master)
            for ten, f in cau_hinh.items()}
    print(bao_cao_tu_bang(bang))


if __name__ == "__main__":
    main()
