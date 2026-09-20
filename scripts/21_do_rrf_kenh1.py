"""
21_do_rrf_kenh1.py — RRF có cộng hưởng được với kênh 1 (SigLIP2) không?

`16_do_rrf.py` đo RRF(2, 4) khi kênh 1 còn mù tiếng Việt (CLIP ViT-B/32,
A10) — không có gì để so, nên chưa từng đưa kênh 1 vào phép đo RRF. Giờ kênh 1
đã sống lại (SigLIP2, A17: 0,3258 — mạnh gấp ~8 lần kênh 4), CLIP ViT-B/32 bị
bỏ hẳn (mù tiếng Việt, không đáng giữ). Script này đo: có kênh nào cộng vào
kênh 1 mà KHÔNG làm tệ đi không, theo đúng kỷ luật CLAUDE.md — mốc nền là
CẤU HÌNH MẠNH NHẤT hiện có (kênh 1 một mình), không phải kênh yếu.

    python scripts/21_do_rrf_kenh1.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay, MOC_DUNG_SAI   # noqa: E402
from dense import KenhAnh                             # noqa: E402
from rrf import hop_nhat                              # noqa: E402

sys.path.insert(0, str(GOC / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("r16", GOC / "scripts" / "16_do_rrf.py")
_r16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r16)
KenhObjects = _r16.KenhObjects


def chong_lan(cau, f1, f2, f4, master) -> pd.DataFrame:
    """RRF chỉ cộng hưởng khi các kênh đề cử CÙNG một row_id. Đo xem có không."""
    dong = []
    for c in cau:
        a, b, d = f1(c), f2(c), f4(c)
        ra, rb, rd = {x.row_id for x in a}, {x.row_id for x in b}, {x.row_id for x in d}
        va, vb, vd = ({x.video_id for x in a}, {x.video_id for x in b},
                     {x.video_id for x in d})
        dong.append({
            "id": c.id,
            "chung_khung_1_2": len(ra & rb), "chung_khung_1_4": len(ra & rd),
            "chung_video_1_2": len(va & vb), "chung_video_1_4": len(va & vd),
        })
    return pd.DataFrame(dong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--moi-video", type=int, default=3)
    a = ap.parse_args()

    cau = tap_dev.doc(a.file)
    master = pd.read_parquet(a.index / "master.parquet")
    print(f"Nạp kênh 1 ({a.matrix})...")
    k1 = KenhAnh(a.index, matrix=a.matrix)
    k2 = KenhVanBan.tu_metadata(master)
    k4 = KenhObjects(a.index, master)

    nho1, nho2, nho4 = {}, {}, {}

    def f1(c):
        if c.id not in nho1:
            nho1[c.id] = k1.tim(c.cau_hoi, k=a.k)
        return nho1[c.id]

    def f2(c):
        if c.id not in nho2:
            nho2[c.id] = k2.tim(c.cau_hoi, k=a.k, moi_video=a.moi_video)
        return nho2[c.id]

    def f4(c):
        if c.id not in nho4:
            nho4[c.id] = k4.tim(c.cau_hoi, k=a.k)
        return nho4[c.id]

    print(f"{len(cau)} câu | k={a.k}\n")

    print("=" * 76)
    print("CHỒNG LẤN với kênh 1 — điều kiện cần để RRF cộng hưởng")
    print("=" * 76)
    cl = chong_lan(cau, f1, f2, f4, master)
    for cot, ten in (("chung_khung_1_2", "1∩2 KHUNG"), ("chung_video_1_2", "1∩2 VIDEO"),
                     ("chung_khung_1_4", "1∩4 KHUNG"), ("chung_video_1_4", "1∩4 VIDEO")):
        print(f"  {ten}: {int((cl[cot] > 0).sum())}/{len(cl)} câu"
              f"   (trung bình {cl[cot].mean():.1f})")
    if cl.chung_khung_1_2.mean() < 1 and cl.chung_khung_1_4.mean() < 1:
        print("\n  ⚠️  Gần như không chung khung nào với kênh 1. Theo bài học A14/A17,\n"
              "      RRF nhiều khả năng chỉ ĐAN XEN chứ không cộng hưởng — dự đoán:\n"
              "      mọi tổ hợp dưới đây sẽ THUA kênh 1 đứng một mình.")

    print("\n" + "=" * 76)
    print("ĐIỂM — mốc nền là kênh 1 MỘT MÌNH (cấu hình mạnh nhất đã biết, A17)")
    print("=" * 76)
    cau_hinh = {
        "kênh 1 SigLIP2 (mốc)": f1,
        "kênh 2 metadata": f2,
        "kênh 4 objects": f4,
        "RRF(1, 2)": lambda c: hop_nhat([f1(c), f2(c)]),
        "RRF(1, 4)": lambda c: hop_nhat([f1(c), f4(c)]),
        "RRF(1, 2, 4)": lambda c: hop_nhat([f1(c), f2(c), f4(c)]),
    }
    # Trọng số hạ dần cho kênh phụ — theo giả thuyết A14: RRF thô coi mọi kênh
    # đáng tin NHƯ NHAU, hạ trọng số kênh yếu phải kéo điểm về gần kênh 1.
    for w in (0.5, 0.2, 0.05):
        cau_hinh[f"RRF(1, 4) trọng số phụ {w}"] = (
            lambda w: lambda c: hop_nhat([f1(c), f4(c)], trong_so=[1.0, w]))(w)
        cau_hinh[f"RRF(1, 2) trọng số phụ {w}"] = (
            lambda w: lambda c: hop_nhat([f1(c), f2(c)], trong_so=[1.0, w]))(w)

    print(bao_cao_do_nhay(cau, cau_hinh, master, MOC_DUNG_SAI, gioi_han=a.k))


if __name__ == "__main__":
    main()
