"""
15_do_bm25.py — Đo kênh 2 (metadata) trên tập dev, và dò các nút của nó.

Đây là kênh ĐẦU TIÊN có thể ra số khác 0 trên tập dev tiếng Việt: metadata là
tiếng Việt, truy vấn là tiếng Việt, không cần model nào dịch ở giữa. Kênh 1
chạy CLIP được 0,0000 (A10) nên mọi cấu hình đo trên nó đều ra "KHÔNG ĐỔI GÌ".

HAI THƯỚC ĐO, VÀ PHẢI ĐỌC ĐÚNG CÁI NÀO
======================================

Metadata mô tả **cả video**, không biết gì về từng khung. Nên báo cáo hai thứ
tách bạch:

    HẠNG VIDEO   video đúng đứng thứ mấy trong 873 video?  <- năng lực THẬT
    ĐIỂM BTC     công thức chấm của giải, trên keyframe    <- đóng góp thật

Điểm BTC của kênh này nhất định thấp, và **thấp không có nghĩa là kênh vô
dụng**: một video trung bình 203 khung, nên dù đoán đúng video ở hạng 1 thì
khung đúng vẫn nằm đâu đó trong 203 khung theo thứ tự thời gian. Giá trị của
kênh nằm ở chỗ hợp nhất — nó thu hẹp còn vài video, kênh ảnh chọn khung.

    python scripts/15_do_bm25.py
    python scripts/15_do_bm25.py --nut      # dò bigram / title x3 / moi_video
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                       # noqa: E402
from bm25 import KenhVanBan                          # noqa: E402
from cham_diem import bao_cao_do_nhay, cham, MOC_DUNG_SAI, tom_tat  # noqa: E402


def hang_video(kenh: KenhVanBan, cau, master) -> pd.DataFrame:
    """Video đúng đứng thứ mấy trong 873 video — thước đo thật của kênh cấp video."""
    dong = []
    for c in cau:
        d = kenh.diem_tai_lieu(c.cau_hoi)
        xep = np.argsort(-d)
        dung = c.video_id(master)
        h = next((i + 1 for i, j in enumerate(xep)
                  if kenh.video_id[j] == dung and d[j] > 0), None)
        dong.append({"id": c.id, "nhom": c.nhom(master), "loai": c.loai,
                     "hang_video": h})
    return pd.DataFrame(dong)


def in_hang_video(d: pd.DataFrame):
    n = len(d)
    co = d.hang_video.notna()
    print(f"  tìm ra video đúng ở đâu đó: {int(co.sum())}/{n} câu "
          f"({co.mean() * 100:.0f}%)")
    for k in (1, 3, 5, 10, 20, 50):
        print(f"    video đúng trong top-{k:<3}: "
              f"{int((d.hang_video <= k).sum()):>3}/{n}  "
              f"({(d.hang_video <= k).mean() * 100:5.1f}%)")
    if co.any():
        print(f"    trung vị hạng (khi tìm ra): {d.hang_video[co].median():.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--nut", action="store_true", help="dò bigram / title×3 / moi_video")
    a = ap.parse_args()

    cau = tap_dev.doc(a.file)
    master = pd.read_parquet(a.index / "master.parquet")
    print(f"{len(cau)} câu | {master.video_id.nunique()} video | "
          f"{len(master):,} keyframe "
          f"({len(master) / master.video_id.nunique():.0f} khung/video)\n")

    kenh = KenhVanBan.tu_metadata(master)

    print("=" * 76)
    print("THƯỚC ĐO 1 — HẠNG VIDEO  (năng lực thật của kênh cấp video)")
    print("=" * 76)
    hv = hang_video(kenh, cau, master)
    in_hang_video(hv)
    print("\n  theo nhóm L:")
    print(hv.assign(top10=hv.hang_video <= 10)
            .groupby("nhom").agg(so_cau=("id", "size"),
                                 tim_ra=("hang_video", lambda s: s.notna().mean()),
                                 top10=("top10", "mean")).round(2).to_string())

    print("\n" + "=" * 76)
    print("THƯỚC ĐO 2 — ĐIỂM BTC  (trên keyframe, công thức chấm của giải)")
    print("=" * 76)
    cau_hinh = {f"moi_video = {mv or 'không giới hạn'}":
                (lambda mv: lambda c: kenh.tim(c.cau_hoi, k=100, moi_video=mv))(mv)
                for mv in (None, 1, 3, 10, 30)}
    print(bao_cao_do_nhay(cau, cau_hinh, master, MOC_DUNG_SAI))

    if not a.nut:
        print("\n(thêm --nut để dò bigram và title×3)")
        return

    print("\n" + "=" * 76)
    print("NÚT — bigram và title×3 có đáng không")
    print("=" * 76)
    khong_bg = KenhVanBan.tu_metadata(master, bigram=False)

    # title×3 tắt: dựng tay, không qua tu_metadata
    g = master.groupby("video_id", sort=True)
    vb, khoa, vids = [], [], []
    for v, sub in g:
        r = sub.iloc[0]
        vb.append(" ".join([str(r.title or ""), str(r.description or ""),
                            str(r.keywords or "")]))
        khoa.append(sub.index.to_numpy(dtype=np.int64))
        vids.append(v)
    khong_t3 = KenhVanBan(vb, khoa, master, ten="metadata")
    khong_t3.video_id = vids

    for ten, k in (("đầy đủ (bigram + title×3)", kenh),
                   ("bỏ bigram", khong_bg),
                   ("bỏ title×3", khong_t3)):
        print(f"\n  {ten}")
        d = hang_video(k, cau, master)
        print(f"    video đúng trong top-10: "
              f"{int((d.hang_video <= 10).sum()):>3}/{len(d)}"
              f"   top-1: {int((d.hang_video <= 1).sum()):>3}/{len(d)}"
              f"   tìm ra: {int(d.hang_video.notna().sum()):>3}/{len(d)}")


if __name__ == "__main__":
    main()
