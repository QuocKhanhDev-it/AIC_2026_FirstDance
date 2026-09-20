"""
21_do_lan_can.py — Đo hai thứ đã viết xong mà CHƯA AI ĐO.

Grep toàn repo trước khi viết script này:

    lan_can.py            không module nào import
    gioi_han_moi_video    không ai gọi

Cái thứ hai đáng nói: **PHẦN C mục 2 gọi nó là ràng buộc CỨNG** — *"mỗi video
≤ 2 slot trong top-5; top-20 trải trên ≥ 8 video"*. Ta tuyên bố bắt buộc rồi
không cài vào đường ống nào. Đây đúng loại lỗ hổng chỉ lộ ra khi đi soát lại.

HAI THỨ ĐO Ở ĐÂY, VÀ VÌ SAO ĐỂ CHUNG MỘT SCRIPT
================================================

Cả hai đều **can thiệp vào danh sách sau khi xếp hạng, trước khi cắt top-K**,
nên chúng cạnh tranh nhau cùng một chỗ trong đường ống và phải so cùng mốc nền.

**`lan_can`** — A8.7 xếp hạng 1 về lợi/công. Câu Q&A của đội AIC'25 cho gợi ý
về *nguyên liệu* còn đáp án nằm ở *bước cắt* vài giây sau: truy hồi đưa tới lân
cận, đi bộ đưa tới đích.

⚠️ Nhưng docstring `lan_can.py` tự cảnh báo: khung lân cận vào với `score = 0`
và **thứ hạng giả**. Nhét thẳng vào là đưa rác lên trên. Nên ở đây chèn chúng
**ngay sau** khung sinh ra chúng, giữ nguyên thứ tự tương đối — cách rẻ nhất
tôn trọng cảnh báo đó mà không cần chấm lại bằng kênh khác.

**`gioi_han_moi_video`** — đổi lại: nó **bỏ bớt**, không thêm vào.

    python scripts/21_do_lan_can.py
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
from lan_can import bien_video, lan_can               # noqa: E402
from objects import KenhObjects                       # noqa: E402
from rrf import gioi_han_moi_video, hop_nhat          # noqa: E402


def no_lan_can(uv: list, master, bien, so_buoc: int, k: int) -> list:
    """Chèn khung lân cận NGAY SAU khung sinh ra chúng.

    Giữ trật tự tương đối của kênh: khung tốt nhất vẫn đứng đầu, lân cận của nó
    đứng ngay sau. Cách này không cần chấm lại mà vẫn không đẩy rác lên trên —
    xem cảnh báo ở `lan_can.lan_can_nhieu`.
    """
    thay, ra = set(), []
    for c in uv:
        if c.row_id not in thay:
            thay.add(c.row_id)
            ra.append(c)
        for x in lan_can(master, c.row_id, so_buoc, bien):
            if x.row_id not in thay:
                thay.add(x.row_id)
                ra.append(x)
        if len(ra) >= k:
            break
    return ra[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)
    bien = bien_video(master)

    # Mốc nền = cấu hình MẠNH NHẤT chạy được không cần model: RRF(objects, OCR)
    # đo được 0,0640 — xem commit kênh 3.
    k4 = KenhObjects(a.index, master)

    p = a.index / "ocr_asr.parquet"
    if not p.exists():
        raise SystemExit(f"Chưa có {p} — cần kênh 3 để dựng mốc nền mạnh nhất.")
    k3 = KenhVanBan.tu_bang_khung(master, pd.read_parquet(p),
                                  cot="text", ten="ocr_asr")
    print(f"{len(cau)} câu | mốc nền RRF(objects, OCR) | "
          f"kênh 3 có {len(k3):,} khung có chữ\n")

    kho = {}

    def moc(c):
        if c.id not in kho:
            kho[c.id] = hop_nhat([k4.tim(c.cau_hoi, k=a.k),
                                  k3.tim(c.cau_hoi, k=a.k)])
        return kho[c.id]

    cau_hinh = {"RRF(objects, OCR) — mốc": moc}
    for b in (2, 5, 10):
        cau_hinh[f"+ lân cận ±{b}"] = (
            lambda b: lambda c: no_lan_can(moc(c), master, bien, b, a.k))(b)
    for mv in (2, 3, 5):
        cau_hinh[f"+ mỗi video ≤ {mv}"] = (
            lambda mv: lambda c: gioi_han_moi_video(moc(c), mv, a.k))(mv)

    print(bao_cao_do_nhay(cau, cau_hinh, master, MOC_DUNG_SAI, gioi_han=a.k))


if __name__ == "__main__":
    main()
