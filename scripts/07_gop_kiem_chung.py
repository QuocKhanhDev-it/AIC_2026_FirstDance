"""
07_gop_kiem_chung.py — Gộp kết quả 02+03 của cả nhóm, tính độ phủ thật.

Mỗi máy chỉ tải được vài nhóm L nên chỉ kiểm chứng được phần của mình.
Script này gom các báo cáo rời thành một con số duy nhất trả lời được câu
"bảng cái đã được chứng minh đúng trên bao nhiêu phần trăm kho?".

Cách dùng:
  1. Người gửi chép hai file ra khỏi index/ của họ:
         index/verify_report.csv        <- script 02 (ảnh keyframe <-> dòng CSV)
         index/verify_clip*.csv         <- script 03 (vector CLIP <-> dòng CSV)
  2. Bỏ vào dev/verify/<nhom_L>/ , giữ nguyên tên verify_report.csv và
     verify_clip.csv.
  3. python scripts/07_gop_kiem_chung.py

KHÔNG cần master.parquet / clip.npy / objects.parquet của người khác:
mỗi bộ 395 MB và giống hệt nhau trừ ba cột đường dẫn tuyệt đối.

Vì sao gộp được: 00_discover.py duyệt video theo thứ tự sorted(video_id) và
01_build_index.py đánh row_id tuần tự, nên row_id <-> (video_id, kf_n) như
nhau trên mọi máy có đủ 873 file CSV. Đã đối chiếu thật 23/23 dòng L29 giữa
hai máy — trùng khít cả kf_n lẫn frame_idx.
"""

import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
VERIFY = GOC / "dev" / "verify"
# 02 dùng KHOP_YEU cho cảnh động: tương quan pixel thấp nhưng vượt xa dòng kề.
DAT_PIXEL = {"KHOP", "KHOP_YEU"}


def doc(ten: str) -> pd.DataFrame:
    khung = []
    for thu_muc in sorted(p for p in VERIFY.iterdir() if p.is_dir()):
        f = thu_muc / ten
        if f.exists():
            d = pd.read_csv(f)
            # Nhóm L lấy từ video_id, KHÔNG lấy từ tên thư mục: một máy giữ
            # nhiều nhóm sẽ xuất chung một file (đo thật: lô L24+L30).
            d["nhom"] = d.video_id.str[:3]
            d["nguon"] = thu_muc.name
            khung.append(d)
    if not khung:
        return pd.DataFrame()
    gop = pd.concat(khung, ignore_index=True)
    trung = gop[gop.duplicated(["video_id", "nhom"], keep=False)]
    if len(trung):
        print(f"!! {trung.video_id.nunique()} video được nhiều nguồn cùng kiểm "
              f"({', '.join(sorted(trung.nguon.unique()))}) — đếm một lần.\n")
    return gop


def main():
    master = GOC / "index" / "master.parquet"
    if not master.exists():
        sys.exit("Thiếu index/master.parquet — chạy 01_build_index.py trước.")
    m = pd.read_parquet(master, columns=["video_id", "fps"])
    m["nhom"] = m.video_id.str[:3]
    tong_video = m.groupby("nhom").video_id.nunique()

    pix, clip = doc("verify_report.csv"), doc("verify_clip.csv")
    if pix.empty and clip.empty:
        sys.exit(f"Chưa có báo cáo nào trong {VERIFY}.")

    print("=" * 72)
    print(f"{'nhóm':<6} {'video':>6} {'02 đã kiểm':>11} {'02 đạt':>8} "
          f"{'03 đã kiểm':>11} {'03 hạng 1':>10}")
    print("-" * 72)
    for nhom in tong_video.index:
        p = pix[pix.nhom == nhom] if not pix.empty else pd.DataFrame()
        c = clip[clip.nhom == nhom] if not clip.empty else pd.DataFrame()
        p_dat = int(p.ket_luan.isin(DAT_PIXEL).sum()) if len(p) else 0
        c_dat = int((c.hang == 1).sum()) if len(c) else 0
        danh = "" if len(p) or len(c) else "   <- chưa ai kiểm"
        print(f"{nhom:<6} {tong_video[nhom]:>6} {len(p):>11} {p_dat:>8} "
              f"{len(c):>11} {c_dat:>10}{danh}")

    n_video = int(tong_video.sum())
    print("-" * 72)
    print(f"{'TỔNG':<6} {n_video:>6} {len(pix):>11} "
          f"{int(pix.ket_luan.isin(DAT_PIXEL).sum()) if len(pix) else 0:>8} "
          f"{len(clip):>11} {int((clip.hang == 1).sum()) if len(clip) else 0:>10}")
    print("=" * 72)

    da_kiem = set()
    for d in (pix, clip):
        if len(d):
            da_kiem |= set(d.video_id)
    print(f"\nĐộ phủ kiểm chứng: {len(da_kiem)}/{n_video} video "
          f"({len(da_kiem)/n_video*100:.1f}%) — "
          f"{len(set(pix.nhom) | set(clip.nhom))}/{len(tong_video)} nhóm L.")

    if len(pix):
        truot = pix[~pix.ket_luan.isin(DAT_PIXEL)]
        if len(truot):
            print(f"\n!! {len(truot)} mẫu TRƯỢT script 02 — phải xem ngay:")
            print(truot[["video_id", "kf_name", "corr", "bien_do",
                         "ket_luan"]].to_string(index=False))
    if len(clip):
        truot = clip[clip.hang != 1]
        if len(truot):
            print(f"\n!! {len(truot)} mẫu KHÔNG hạng 1 ở script 03:")
            print(truot[["video_id", "kf_n", "cosine", "hang",
                         "cach_biet"]].to_string(index=False))
            print("   (hạng 2 do keyframe TRÙNG LẶP thì bình thường; hạng "
                  "xa hơn mới là lệch hàng clip.npy)")

    # fps lạ là cái bẫy nguy hiểm nhất còn lại: mọi phép quy đổi giây <-> frame
    # đều sai nếu 26.44 hoặc 29.97 bị làm tròn thành 25/30 ở đâu đó.
    la = m[~m.fps.isin([25.0, 30.0])].groupby(["nhom", "fps"]).video_id.nunique()
    if len(la):
        print("\nVIDEO fps LẠ (bắt buộc phải có người kiểm chứng):")
        for (nhom, fps), sl in la.items():
            xong = len(da_kiem & set(m[(m.nhom == nhom) & (m.fps == fps)].video_id))
            dau = "OK" if xong else "CHƯA AI KIỂM"
            print(f"  {nhom}  fps={fps:<6} {sl:>3} video   đã kiểm {xong:>3}  <- {dau}")


if __name__ == "__main__":
    main()
