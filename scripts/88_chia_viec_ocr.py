"""
88_chia_viec_ocr.py — Chia việc chạy VietOCR, TÁCH RIÊNG L26.

    python scripts/88_chia_viec_ocr.py            # xem trước
    python scripts/88_chia_viec_ocr.py --ghi      # ghi ra chia_ocr/

VÌ SAO CHIA MỚI CHỨ KHÔNG DÙNG LẠI `chia_caption/`

`chia_caption` trải L26 đều ra cả 12 phần — **mỗi phần có 39–43 video L26 trên
~73 video**, tức 55–59%. Ai chưa tải được dataset L26 thì không chạy nổi phần
nào cả.

Bài học "ĐỪNG CHIA LẠI KHI ĐÃ CÓ NGƯỜI CHẠY" (đợt caption: chia lại làm ba phần
hợp lệ bị đánh dấu nhầm là sai phần) nói về bản chia **đang có người chạy dở**.
OCR chưa ai chạy phần nào, nên chia riêng ở đây là hợp lệ — và sau khi ghi ra
thì áp dụng đúng bài học đó: **không chia lại nữa**.

CÂN THEO SỐ KHUNG, KHÔNG THEO SỐ VIDEO

Số khung mỗi video lệch rất mạnh: L23 có 25 video / 2.326 khung, còn L25 có 88
video / 37.445 khung. Chia đều theo *video* thì phần nào rơi vào L25 sẽ chạy
lâu gấp nhiều lần. Xếp thùng tham lam theo *khung* cho các phần lệch dưới 5%.

    L26        : 498 video,  79.590 khung, 16,4 giờ  (44,9% kho)
    KHÔNG L26  : 375 video,  97.731 khung, 20,1 giờ

Ở 0,74 s/ảnh (đo thật trên T4, xem A75): 7 phần A + 5 phần B ≈ 3 giờ/phần,
vừa khít phiên 6 tiếng với dư địa gấp đôi.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

GIAY_MOI_ANH = 0.74                                   # đo thật trên T4


def xep_thung(muc: list, so_thung: int) -> list[list]:
    """Xếp thùng tham lam: video nặng nhất vào thùng đang nhẹ nhất."""
    thung = [[] for _ in range(so_thung)]
    nang = [0] * so_thung
    for v, n in sorted(muc, key=lambda x: -x[1]):
        i = nang.index(min(nang))
        thung[i].append(v)
        nang[i] += n
    return thung


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--ra", default=GOC / "chia_ocr", type=Path)
    ap.add_argument("--phan-a", type=int, default=7, help="số phần KHÔNG L26")
    ap.add_argument("--phan-b", type=int, default=5, help="số phần CHỈ L26")
    ap.add_argument("--ghi", action="store_true")
    a = ap.parse_args()

    m = pd.read_parquet(a.index / "master.parquet", columns=["video_id"])
    dem = m.video_id.value_counts().to_dict()

    ngoai = [(v, n) for v, n in dem.items() if not v.startswith("L26")]
    l26 = [(v, n) for v, n in dem.items() if v.startswith("L26")]
    print(f"KHÔNG L26 : {len(ngoai):>4} video  {sum(n for _, n in ngoai):>7,} khung")
    print(f"CHỈ L26   : {len(l26):>4} video  {sum(n for _, n in l26):>7,} khung\n")

    nhom = {"A": xep_thung(ngoai, a.phan_a), "B": xep_thung(l26, a.phan_b)}

    print(f"{'phần':>6}{'video':>7}{'khung':>9}{'giờ GPU':>10}   cần L26")
    print("-" * 48)
    for ten, cac in nhom.items():
        for i, vs in enumerate(cac, 1):
            k = sum(dem[v] for v in vs)
            print(f"{ten + str(i):>6}{len(vs):>7}{k:>9,}"
                  f"{k * GIAY_MOI_ANH / 3600:>10.1f}"
                  f"{'   KHÔNG' if ten == 'A' else '   có':>10}")

    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi thật)")
        return

    if a.ra.exists() and any(a.ra.iterdir()):
        raise SystemExit(
            f"\n❌ {a.ra} đã có nội dung. ĐỪNG CHIA LẠI khi đã có người chạy —\n"
            f"   xoá tay nếu chắc chắn chưa ai bắt đầu.")
    a.ra.mkdir(parents=True, exist_ok=True)
    for ten, cac in nhom.items():
        for i, vs in enumerate(cac, 1):
            f = a.ra / f"phan_{ten}{i}.txt"
            f.write_text("\n".join(sorted(vs)) + "\n", encoding="utf-8",
                         newline="\n")
    print(f"\n✅ {a.phan_a + a.phan_b} file trong {a.ra}")
    print(f"   phần A1–A{a.phan_a}: KHÔNG cần L26")
    print(f"   phần B1–B{a.phan_b}: chỉ L26")


if __name__ == "__main__":
    main()
