"""
17_compare_tu_metadata_bugs.py — Hai thay đổi của `tu_metadata()`, tách riêng.

Bản gốc (NgThanhDat-ne) so V0 → V1 → V2 và tìm ra một lỗi THẬT trong `tu_metadata`:
ghép các trường bằng `" "` khiến `tach()` sinh **bigram lai bắc cầu qua biên
trường** — `"…VIVU TV" + "VIVU TV"` đẻ ra `tv_vivu`, một cụm không có trong văn
bản gốc. Lỗi đó là của tôi, và sửa là đúng.

VÌ SAO VIẾT LẠI SCRIPT NÀY

V1 đổi **hai thứ cùng lúc** — vừa dọn rác (URL/hashtag/mention) vừa đổi cách
ghép trường — nên bảng V0/V1/V2 không nói được **thứ nào** tạo ra khác biệt.
Cùng cái bẫy đã vấp ở A14.1 khi đo dedup, và ở đó nó suýt làm ta quy nhầm công
cho hợp nhất.

Ở đây dò **2×2**:

    ghép trường bằng   " "  hoặc  ". "
    dọn rác            tắt  hoặc  bật

Và chấm bằng `bao_cao_do_nhay` chứ không chỉ đếm top-k: 97 câu thì lệch 1–3 câu
nằm gọn trong nhiễu, mà bảng đếm thô không cho biết ngưỡng nhiễu ở đâu.

    python scripts/17_compare_tu_metadata_bugs.py
    python scripts/17_compare_tu_metadata_bugs.py --minh-hoa   # ví dụ bigram lai
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan, don_metadata, tach, _RAC  # noqa: E402
from cham_diem import bao_cao_do_nhay, MOC_DUNG_SAI   # noqa: E402


def dung(master: pd.DataFrame, ghep: str, don_rac: bool,
         title_lap: int = 3) -> KenhVanBan:
    """Dựng kênh metadata với một tổ hợp nút cụ thể."""
    vids, van_ban, khoa = [], [], []
    for v, sub in master.groupby("video_id", sort=True):
        r = sub.iloc[0]
        t = str(r.title or "")
        d = str(r.description or "")
        w = str(r.keywords or "")
        if don_rac:
            # Dọn bằng CHÍNH ký tự đang dùng để ghép, để hai nút độc lập nhau.
            d, w = _RAC.sub(ghep, d), _RAC.sub(ghep, w)
        van_ban.append(ghep.join([t] * title_lap + [d, w]))
        khoa.append(sub.index.to_numpy(dtype=np.int64))
        vids.append(v)
    k = KenhVanBan(van_ban, khoa, master, ten="metadata")
    k.video_id = vids
    return k


def hang_video(kenh: KenhVanBan, cau, master) -> pd.Series:
    """Hạng của video đúng, NaN nếu không tìm ra."""
    ra = []
    for c in cau:
        d = kenh.diem_tai_lieu(c.cau_hoi)
        xep = np.argsort(-d)
        v = c.video_id(master)
        ra.append(next((i + 1 for i, j in enumerate(xep)
                        if kenh.video_id[j] == v and d[j] > 0), np.nan))
    return pd.Series(ra, index=[c.id for c in cau])


def minh_hoa():
    """Bigram lai — cả hai kiểu: bắc cầu qua RÁC, và bắc cầu qua BIÊN TRƯỜNG."""
    s = "Chi tiết tại https://youtu.be/xyz123 #nauan @chef nấu ăn ngon nhé"
    print("Bắc cầu qua RÁC")
    print(f'  "{s}"')
    print(f"  dọn bằng ' '  -> 'tại_nấu' có mặt: {'tại_nấu' in tach(_RAC.sub(' ', s))}")
    print(f"  dọn bằng '. ' -> 'tại_nấu' có mặt: {'tại_nấu' in tach(don_metadata(s))}")

    # ⚠️ Token đúng là `tv_món` (cuối lần lặp này nối đầu lần lặp sau), KHÔNG
    # phải `tv_vivu`. Bản đầu tôi kiểm nhầm `tv_vivu` nên minh hoạ ra False ở
    # cả hai cột và trông như lỗi không tồn tại — lỗi CÓ THẬT.
    t = "MÓN NGON MỖI NGÀY VIVU TV"
    print("\nBắc cầu qua BIÊN TRƯỜNG (title lặp 3 lần)")
    print(f'  title = "{t}"')
    for ghep, ten in ((" ", "' '"), (". ", "'. '")):
        tok = tach(ghep.join([t] * 3))
        print(f"  ghép bằng {ten:<5} -> 'tv_món' có mặt: {'tv_món' in tok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--minh-hoa", action="store_true")
    a = ap.parse_args()

    if a.minh_hoa:
        return minh_hoa()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)
    print(f"{master.video_id.nunique()} video | {len(cau)} câu dev\n")

    nut = {
        "V0  ghép ' '  · rác giữ ": (" ", False),
        "Va  ghép '. ' · rác giữ ": (". ", False),
        "Vb  ghép ' '  · rác dọn ": (" ", True),
        "V2  ghép '. ' · rác dọn ": (". ", True),
    }
    kenh = {ten: dung(master, g, d) for ten, (g, d) in nut.items()}
    hv = {ten: hang_video(k, cau, master) for ten, k in kenh.items()}

    print("=" * 84)
    print("2×2 — nút nào thật sự làm việc")
    print("=" * 84)
    print(f"  {'cấu hình':<26}{'từ vựng':>9}{'top-1':>7}{'top-10':>8}"
          f"{'top-20':>8}{'tìm ra':>8}{'trung vị':>10}")
    print("  " + "-" * 76)
    for ten, k in kenh.items():
        h = hv[ten]
        print(f"  {ten:<26}{len(k.co_dau.chi_muc):>9,}"
              f"{int((h <= 1).sum()):>7}{int((h <= 10).sum()):>8}"
              f"{int((h <= 20).sum()):>8}{int(h.notna().sum()):>8}"
              f"{h[h.notna()].median():>10.0f}")

    print("\n  So theo cặp trên HẠNG VIDEO (không tìm ra = hạng 9999):")
    moc = hv["V0  ghép ' '  · rác giữ "].fillna(9999)
    for ten in list(nut)[1:]:
        h = hv[ten].fillna(9999)
        d = moc - h                      # dương = hạng tốt lên
        print(f"    {ten:<26} tốt lên {int((d > 0).sum()):>3} · "
              f"kém đi {int((d < 0).sum()):>3} · như cũ {int((d == 0).sum()):>3}")

    print("\n" + "=" * 84)
    print("ĐIỂM BTC — thước đo thật sự quyết định")
    print("=" * 84)
    print(bao_cao_do_nhay(cau, {
        ten: (lambda k: lambda c: k.tim(c.cau_hoi, k=100, moi_video=3))(k)
        for ten, k in kenh.items()
    }, master, MOC_DUNG_SAI))


if __name__ == "__main__":
    main()
