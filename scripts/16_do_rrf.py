"""
16_do_rrf.py — RRF có thật sự cộng hưởng không, đo trên hai kênh ĐANG CHẠY.

Cho tới nay RRF chưa bao giờ đo được: nó cần ≥ 2 kênh chạy được từ truy vấn
tiếng Việt, mà kênh 1 chạy CLIP thì được 0,0000 (A10). Nay có đủ hai —
kênh 2 (metadata, A12) và kênh 4 (objects + bảng nhãn tiếng Việt) — nên câu hỏi
này trả lời được, không cần chờ máy GPU.

CÂU HỎI THẬT SỰ CẦN TRẢ LỜI

RRF cộng `1/(k + hạng)` từ mỗi kênh. Một ứng viên chỉ ĐƯỢC LỢI khi **nhiều kênh
cùng đề cử NÓ** — cùng một `row_id`. Nếu hai danh sách không giao nhau chỗ nào
thì RRF không cộng hưởng được gì, nó chỉ **đan xen** hai danh sách lại, và kết
quả có thể TỆ HƠN kênh tốt hơn khi đứng một mình: mỗi ứng viên tốt bị đẩy lùi
một bậc bởi một ứng viên của kênh kia.

Mà ở đây có lý do để nghi ngờ: kênh 2 là kênh **cấp video** (trả về mọi khung
của video khớp), kênh 4 là kênh **cấp khung**. Hai kênh nói hai thứ ngôn ngữ
khác nhau về độ mịn — giao nhau đúng `row_id` là chuyện hiếm.

Nên script này đo cả **độ chồng lấn**, không chỉ đo điểm. Điểm nói RRF hơn hay
kém; chồng lấn nói VÌ SAO, và sửa ở đâu.

    python scripts/16_do_rrf.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay, MOC_DUNG_SAI   # noqa: E402
from objects import (load_channel, nap_bang_nhan,     # noqa: E402
                     nhan_tu_truy_van, object_score)
from rrf import hop_nhat, hop_nhat_hai_tang           # noqa: E402
from schema import Candidate                          # noqa: E402


class KenhObjects:
    """Bọc `objects.py` thành kênh trả `list[Candidate]`, cho đồng dạng với
    `KenhAnh` và `KenhVanBan` — RRF không cần biết bên trong kênh nào là gì."""

    def __init__(self, index_dir, master):
        self.master = master
        self.ch = load_channel(index_dir)
        self.bang = nap_bang_nhan()
        self.rid = master.row_id.values

    def nhan(self, cau: str) -> list[str]:
        return nhan_tu_truy_van(cau, self.bang)

    def tim(self, cau: str, k=100, moi_video=None) -> list[Candidate]:
        nhan = self.nhan(cau)
        if not nhan:
            return []                    # không rút ra nhãn nào -> im lặng, đừng bịa
        d = object_score(self.rid, nhan, self.ch)
        top = np.argsort(-d)[:k * (moi_video or 1) + 200]
        m, ra, dem = self.master, [], {}
        for i in top:
            if d[i] <= 0:
                break
            v = m.video_id.iloc[i]
            if moi_video and dem.get(v, 0) >= moi_video:
                continue
            dem[v] = dem.get(v, 0) + 1
            ra.append(Candidate(row_id=int(i), video_id=v,
                                frame_idx=int(m.frame_idx.iloc[i]),
                                score=float(d[i]), source="objects",
                                meta={"pts_time": float(m.pts_time.iloc[i])}))
            if len(ra) >= k:
                break
        return ra


def chong_lan(cau, f2, f4, master) -> pd.DataFrame:
    """RRF chỉ cộng hưởng khi hai kênh đề cử CÙNG một `row_id`. Đo xem có không."""
    dong = []
    for c in cau:
        a, b = f2(c), f4(c)
        ra, rb = {x.row_id for x in a}, {x.row_id for x in b}
        va, vb = {x.video_id for x in a}, {x.video_id for x in b}
        dong.append({
            "id": c.id,
            "n2": len(a), "n4": len(b),
            "chung_khung": len(ra & rb),
            "chung_video": len(va & vb),
            "co_nhan": bool(b),
        })
    return pd.DataFrame(dong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--moi-video", type=int, default=3,
                    help="giới hạn cho kênh 2 — không có thì một video khớp "
                         "chiếm sạch 100 chỗ")
    a = ap.parse_args()

    cau = tap_dev.doc(a.file)
    master = pd.read_parquet(a.index / "master.parquet")
    k2 = KenhVanBan.tu_metadata(master)
    k4 = KenhObjects(a.index, master)

    nho2, nho4 = {}, {}

    def f2(c):
        if c.id not in nho2:
            nho2[c.id] = k2.tim(c.cau_hoi, k=a.k, moi_video=a.moi_video)
        return nho2[c.id]

    def f4(c):
        if c.id not in nho4:
            nho4[c.id] = k4.tim(c.cau_hoi, k=a.k)
        return nho4[c.id]

    print(f"{len(cau)} câu | kênh 2 moi_video={a.moi_video} | k={a.k}\n")

    print("=" * 76)
    print("CHỒNG LẤN — điều kiện cần để RRF cộng hưởng được")
    print("=" * 76)
    cl = chong_lan(cau, f2, f4, master)
    print(f"  kênh 4 rút ra được nhãn: {int(cl.co_nhan.sum())}/{len(cl)} câu")
    print(f"  hai kênh chung >= 1 KHUNG: {int((cl.chung_khung > 0).sum())}/{len(cl)} câu"
          f"   (trung bình {cl.chung_khung.mean():.1f} khung)")
    print(f"  hai kênh chung >= 1 VIDEO: {int((cl.chung_video > 0).sum())}/{len(cl)} câu"
          f"   (trung bình {cl.chung_video.mean():.1f} video)")
    if cl.chung_khung.mean() < 1:
        print("\n  ⚠️  Gần như không chung khung nào. RRF sẽ chỉ ĐAN XEN hai danh\n"
              "      sách chứ không cộng hưởng — và đan xen thì đẩy lùi ứng viên\n"
              "      tốt của kênh mạnh, tức có thể TỆ HƠN kênh mạnh đứng một mình.")

    print("\n" + "=" * 76)
    print("ĐIỂM")
    print("=" * 76)
    # Mốc nền phải là KÊNH MẠNH NHẤT, không phải kênh tiện tay đặt trước. Câu
    # hỏi cần trả lời là "hợp nhất có hơn kênh mạnh nhất không" — so với kênh
    # yếu thì RRF gần như luôn thắng, mà thắng như vậy chẳng nói lên điều gì.
    cau_hinh = {
        "kênh 4 objects (mốc)": f4,
        "kênh 2 metadata": f2,
        "RRF thô (2, 4)": lambda c: hop_nhat([f2(c), f4(c)]),
    }
    for mv in (1, 3, 10):
        cau_hinh[f"RRF 2 tầng, mỗi video {mv}"] = (
            lambda mv: lambda c: hop_nhat_hai_tang(
                [f2(c), f4(c)], moi_video=mv, gioi_han=a.k))(mv)
    # Kênh 4 đứng một mình nhưng đi qua tầng 2: tách đóng góp của TẦNG 1 ra
    # khỏi đóng góp của việc chỉ đơn thuần rải đều theo video. Không có dòng
    # này thì mọi cải thiện đều dễ bị quy nhầm cho "hợp nhất".
    cau_hinh["chỉ kênh 4, qua 2 tầng"] = lambda c: hop_nhat_hai_tang(
        [f4(c)], moi_video=3, gioi_han=a.k)

    # TRỌNG SỐ — kiểm chính lời giải thích cho A14.
    #
    # Giả thuyết: RRF làm tệ đi KHÔNG phải vì "thiếu kênh", mà vì nó coi mọi
    # kênh đáng tin NHƯ NHAU. Ứng viên hạng 1 của một kênh chết vẫn được cộng
    # `1/(60+1)` — đúng bằng ứng viên hạng 1 của kênh tốt. Nếu đúng vậy thì hạ
    # trọng số kênh yếu phải kéo điểm về sát kênh mạnh đứng một mình.
    for w in (0.5, 0.2, 0.05):
        cau_hinh[f"RRF(2, 4) trọng số {w} : 1"] = (
            lambda w: lambda c: hop_nhat([f2(c), f4(c)], trong_so=[w, 1.0]))(w)

    print(bao_cao_do_nhay(cau, cau_hinh, master, MOC_DUNG_SAI, gioi_han=a.k))


if __name__ == "__main__":
    main()
