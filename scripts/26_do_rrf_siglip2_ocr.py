"""
26_do_rrf_siglip2_ocr.py — RRF(SigLIP2, OCR+ASR mới): ứng viên mạnh nhất
chưa ai thử.

VÌ SAO SCRIPT NÀY, VÀ VÌ SAO CHƯA AI ĐO ĐƯỢC TỚI GIỜ
=====================================================

`21_do_rrf_kenh1.py` đo RRF(kênh 1, kênh 2/kênh 4) nhưng KHÔNG có kênh 3 —
lúc viết, `ocr_asr.parquet` mới phủ 26,5% (A13/A21). A25 nạp dữ liệu
OCR+ASR mới: kênh 3 nhảy lên **93,2%/77,4%**, điểm một mình 0,1183 — hơn cả
RRF(objects, OCR) cũ. Khoảng cách với kênh 1 SigLIP2 (0,3258, A17) thu từ
8 lần (A14.2, lý do dìm objects xuống 0,3) xuống còn **2,8 lần** — đúng vùng
A14.2 nói RRF bắt đầu có lãi. Chưa ai đo RRF(1, 3) trên dữ liệu OCR mới.

    python scripts/26_do_rrf_siglip2_ocr.py --cache index/truy_van.npz

Có `--cache` thì KHÔNG nạp model SigLIP2 — dùng vector đã mã hoá sẵn bằng
`scripts/25_ma_hoa_truy_van.py --tap-dev`, chạy được trên máy RAM thấp.
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
from dense import KenhAnh, KenhAnhCache               # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def chong_lan(cau, f1, f3, master) -> pd.DataFrame:
    """RRF chỉ cộng hưởng khi hai kênh đề cử CÙNG một row_id/video — đo có
    không, theo đúng cách 21_do_rrf_kenh1.py đã làm cho kênh 2/4."""
    dong = []
    for c in cau:
        a, b = f1(c), f3(c)
        ra, rb = {x.row_id for x in a}, {x.row_id for x in b}
        va, vb = {x.video_id for x in a}, {x.video_id for x in b}
        dong.append({"id": c.id, "chung_khung": len(ra & rb),
                     "chung_video": len(va & vb)})
    return pd.DataFrame(dong)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--cache", default=None, metavar="FILE.npz",
                    help="vector truy vấn đã mã hoá sẵn — không nạp model "
                         "(scripts/25_ma_hoa_truy_van.py --tap-dev)")
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    p = a.index / "ocr_asr.parquet"
    if not p.exists():
        raise SystemExit(f"Chưa có {p} — cần kênh 3.")
    b = pd.read_parquet(p)
    if b.get("text", pd.Series(dtype=str)).fillna("").str.strip().eq("").all():
        raise SystemExit(f"{p} không có dòng nào có chữ.")
    k3 = KenhVanBan.tu_bang_khung(master, b, cot="text", ten="ocr_asr")
    print(f"kênh 3: OCR/ASR, {len(k3):,} khung có chữ ({p.name})")

    if a.cache:
        k1 = KenhAnhCache(a.index, a.cache, matrix=a.matrix)
        thieu = k1.co_du([c.cau_hoi for c in cau])
        if thieu:
            raise SystemExit(
                f"\n❌ {len(thieu)} câu chưa có trong cache, ví dụ:\n"
                f"  {thieu[0][:100]!r}\n\n"
                f"  Mã hoá thêm: python scripts/25_ma_hoa_truy_van.py "
                f"--tap-dev --gop")
        print(f"kênh 1: {a.matrix} — TỪ CACHE {Path(a.cache).name}, "
              f"không nạp model")
    else:
        print(f"Nạp kênh 1 ({a.matrix})...")
        k1 = KenhAnh(a.index, matrix=a.matrix)

    nho1, nho3 = {}, {}

    def f1(c):
        if c.id not in nho1:
            nho1[c.id] = k1.tim(c.cau_hoi, k=a.k)
        return nho1[c.id]

    def f3(c):
        if c.id not in nho3:
            nho3[c.id] = k3.tim(c.cau_hoi, k=a.k)
        return nho3[c.id]

    print(f"\n{len(cau)} câu | k={a.k}\n")

    print("=" * 76)
    print("CHỒNG LẤN với kênh 1 — điều kiện cần để RRF cộng hưởng")
    print("=" * 76)
    cl = chong_lan(cau, f1, f3, master)
    for cot, ten in (("chung_khung", "1∩3 KHUNG"), ("chung_video", "1∩3 VIDEO")):
        print(f"  {ten}: {int((cl[cot] > 0).sum())}/{len(cl)} câu"
              f"   (trung bình {cl[cot].mean():.1f})")
    if cl.chung_khung.mean() < 1:
        print("\n  ⚠️  Gần như không chung khung nào với kênh 1. Theo A14/A17,\n"
              "      RRF nhiều khả năng chỉ ĐAN XEN chứ không cộng hưởng.")

    print("\n" + "=" * 76)
    print("ĐIỂM — mốc nền là kênh 1 MỘT MÌNH (cấu hình mạnh nhất đã biết, A17)")
    print("=" * 76)
    cau_hinh = {
        "kênh 1 SigLIP2 (mốc)": f1,
        "kênh 3 OCR/ASR mới": f3,
        "RRF(1, 3) trọng số 1,0 : 1,0": lambda c: hop_nhat([f1(c), f3(c)]),
    }
    for w in (0.5, 0.3, 0.2, 0.1):
        cau_hinh[f"RRF(1, 3) trọng số phụ {w}"] = (
            lambda w: lambda c: hop_nhat([f1(c), f3(c)], trong_so=[1.0, w]))(w)

    print(bao_cao_do_nhay(cau, cau_hinh, master, MOC_DUNG_SAI, gioi_han=a.k))


if __name__ == "__main__":
    main()
