"""
compare_tu_metadata_bugs.py — So sánh chi tiết các phiên bản tu_metadata()

Khảo sát 3 phiên bản:
  V0 (Gốc):             Ghép " " + KHÔNG dọn rác
  V1 (Thay " "):        Ghép ". " + Dọn rác bằng " " (vẫn dính spurious bigrams khi rác nằm giữa câu, vd: 'tại_nấu')
  V2 (Thay ". " Mới):   Ghép ". " + Dọn rác bằng ". " (rác đóng vai trò cum-breaker, triệt tiêu 100% spurious bigrams)
"""

import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev
from bm25 import KenhVanBan, tach, _RAC

def build_v0_raw(master: pd.DataFrame) -> KenhVanBan:
    """V0 Gốc: ghép ' ' và không dọn rác."""
    g = master.groupby("video_id", sort=True)
    vids, van_ban, khoa = [], [], []
    for v, sub in g:
        r = sub.iloc[0]
        t = str(r.title or "")
        van_ban.append(" ".join([t, t, t, str(r.description or ""), str(r.keywords or "")]))
        khoa.append(sub.index.to_numpy(dtype=np.int64))
        vids.append(v)
    k = KenhVanBan(van_ban, khoa, master, ten="V0_Goc")
    k.video_id = vids
    return k

def build_v1_space(master: pd.DataFrame) -> KenhVanBan:
    """V1: ghép '. ' và dọn rác bằng ' '."""
    g = master.groupby("video_id", sort=True)
    vids, van_ban, khoa = [], [], []
    for v, sub in g:
        r = sub.iloc[0]
        t = str(r.title or "")
        desc = _RAC.sub(" ", str(r.description or ""))
        kw = _RAC.sub(" ", str(r.keywords or ""))
        van_ban.append(". ".join([t, t, t, desc, kw]))
        khoa.append(sub.index.to_numpy(dtype=np.int64))
        vids.append(v)
    k = KenhVanBan(van_ban, khoa, master, ten="V1_Space")
    k.video_id = vids
    return k

def build_v2_dot(master: pd.DataFrame) -> KenhVanBan:
    """V2 Mới: ghép '. ' và dọn rác bằng '. ' (dùng KenhVanBan.tu_metadata hiện tại)."""
    return KenhVanBan.tu_metadata(master)

def hang_video(kenh: KenhVanBan, cau, master) -> pd.DataFrame:
    dong = []
    for c in cau:
        d = kenh.diem_tai_lieu(c.cau_hoi)
        xep = np.argsort(-d)
        dung = c.video_id(master)
        h = next((i + 1 for i, j in enumerate(xep)
                  if kenh.video_id[j] == dung and d[j] > 0), None)
        score_dung = float(d[next((j for j in range(len(kenh.video_id)) if kenh.video_id[j] == dung), -1)]) if dung else 0.0
        dong.append({
            "id": c.id,
            "cau_hoi": c.cau_hoi,
            "nhom": c.nhom(master),
            "loai": c.loai,
            "video_dung": dung,
            "hang_video": h,
            "score_dung": score_dung
        })
    return pd.DataFrame(dong)

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    master = pd.read_parquet(GOC / "index" / "master.parquet")
    cau = tap_dev.doc(GOC / "dev" / "tap_dev.jsonl")

    print("=" * 85)
    print("SO SÁNH: V0 (Gốc) vs V1 (Dọn rác bằng ' ') vs V2 (Dọn rác bằng '. ')")
    print(f"Tổng số video: {master.video_id.nunique()} | Số câu dev: {len(cau)}")
    print("=" * 85)

    # 1. Khảo sát ví dụ thực tế về spurious bigram khi thay bằng ' ' vs '. '
    vi_du = "Chi tiết tại https://youtu.be/xyz123 #nauan @chef nấu ăn ngon nhé"
    t_v1 = tach(_RAC.sub(" ", vi_du))
    t_v2 = tach(_RAC.sub(". ", vi_du))
    print("\n--- [1] MINH HỌA LỖI SPURIOUS BIGRAM KHI RÁC Ở GIỮA CÂU ---")
    print(f"  Văn bản mẫu: \"{vi_du}\"")
    print(f"  V1 (thay ' '):  Token sinh ra = {t_v1}")
    print(f"     => 'tại_nấu' xuất hiện: {'tại_nấu' in t_v1} (LỖI: bắc cầu qua rác!)")
    print(f"  V2 (thay '. '): Token sinh ra = {t_v2}")
    print(f"     => 'tại_nấu' xuất hiện: {'tại_nấu' in t_v2} (CHUẨN: ngắt cụm thành công)")

    # 2. Xây dựng 3 chỉ mục
    k0 = build_v0_raw(master)
    k1 = build_v1_space(master)
    k2 = build_v2_dot(master)

    tok0 = set(k0.co_dau.chi_muc.keys())
    tok1 = set(k1.co_dau.chi_muc.keys())
    tok2 = set(k2.co_dau.chi_muc.keys())

    # Tìm những bigram có trong V1 nhưng đã bị V2 loại bỏ vì là bigram lai bắc cầu qua URL/hashtag/mention
    spurious_in_v1 = tok1 - tok2
    print(f"\n--- [2] THỐNG KÊ TOÀN BỘ CHỈ MỤC TRÊN 873 VIDEO ---")
    print(f"{'Thông số':<32} {'V0 (Gốc)':>16} {'V1 (Thay Space)':>16} {'V2 (Thay Dot)':>16}")
    print("-" * 85)
    print(f"{'Từ vựng có dấu (Vocab size)':<32} {len(tok0):>16,d} {len(tok1):>16,d} {len(tok2):>16,d}")
    print(f"{'Từ vựng không dấu':<32} {len(k0.khong_dau.chi_muc):>16,d} {len(k1.khong_dau.chi_muc):>16,d} {len(k2.khong_dau.chi_muc):>16,d}")
    print(f"{'Tổng token có dấu':<32} {int(k0.co_dau.dai.sum()):>16,d} {int(k1.co_dau.dai.sum()):>16,d} {int(k2.co_dau.dai.sum()):>16,d}")
    print(f"{'Độ dài TB dai_tb':<32} {k0.co_dau.dai_tb:>16.2f} {k1.co_dau.dai_tb:>16.2f} {k2.co_dau.dai_tb:>16.2f}")

    print(f"\n--- [3] PHÁT HIỆN SPURIOUS BIGRAMS DO V1 TẠO RA (V2 ĐÃ TRIỆT TIÊU) ---")
    print(f"  Tổng số spurious bigrams trong V1 bị V2 loại bỏ: {len(spurious_in_v1):,} bigrams")
    sample_spurious = sorted([b for b in spurious_in_v1 if "_" in b])
    print(f"  Ví dụ 20 bigram lai qua rác thực tế trong kho video:")
    for b in sample_spurious[:20]:
        print(f"    • {b}")

    # 3. Đánh giá chất lượng trên tập Dev
    hv0 = hang_video(k0, cau, master)
    hv1 = hang_video(k1, cau, master)
    hv2 = hang_video(k2, cau, master)

    print(f"\n--- [4] ĐÁNH GIÁ TRÊN TẬP DEV (97 CÂU HỎI) ---")
    print(f"{'Thước đo Hạng Video':<32} {'V0 (Gốc)':>16} {'V1 (Thay Space)':>16} {'V2 (Thay Dot)':>16}")
    print("-" * 85)
    for top_k in (1, 3, 5, 10, 20, 50):
        c0 = int((hv0.hang_video <= top_k).sum())
        c1 = int((hv1.hang_video <= top_k).sum())
        c2 = int((hv2.hang_video <= top_k).sum())
        print(f"Top-{top_k:<2} video đúng: {c0:>11}/{len(cau)} {c1:>11}/{len(cau)} {c2:>11}/{len(cau)}")

    co0 = hv0.hang_video.notna()
    co1 = hv1.hang_video.notna()
    co2 = hv2.hang_video.notna()
    print(f"{'Tìm ra video (>0 điểm)':<32} {int(co0.sum()):>11}/{len(cau)} {int(co1.sum()):>11}/{len(cau)} {int(co2.sum()):>11}/{len(cau)}")
    print(f"{'Trung vị hạng (khi tìm ra)':<32} {hv0.hang_video[co0].median():>16.1f} {hv1.hang_video[co1].median():>16.1f} {hv2.hang_video[co2].median():>16.1f}")

    # So sánh V2 vs V1 theo cặp
    diff_v2_v1 = []
    for i in range(len(cau)):
        h1 = hv1.iloc[i].hang_video
        h2 = hv2.iloc[i].hang_video
        r1 = 9999 if pd.isna(h1) else h1
        r2 = 9999 if pd.isna(h2) else h2
        diff_v2_v1.append(r1 - r2)
    diff_v2_v1 = np.array(diff_v2_v1)
    
    print(f"\n  So sánh trực tiếp V2 (. ) vs V1 ( ):")
    print(f"  • Thắng (hạng tốt hơn): {(diff_v2_v1 > 0).sum()} câu")
    print(f"  • Thua (hạng kém hơn):  {(diff_v2_v1 < 0).sum()} câu")
    print(f"  • Hòa (bằng nhau):      {(diff_v2_v1 == 0).sum()} câu")

if __name__ == "__main__":
    main()
