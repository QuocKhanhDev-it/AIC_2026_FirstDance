"""
32_dung_tu_dap_an_tay.py — Dựng bài nộp từ ĐÁP ÁN NGƯỜI ĐÃ SOÁT TAY.

Khác mọi script khác trong `scripts/`: ở đây **không có truy hồi nào cả**. Nhóm
đã mở ảnh xem tận mắt và chốt `video_id` + số keyframe cho từng gói; việc của
script là đổi số keyframe thành `frame_idx` và đóng gói đúng định dạng BTC.

    python scripts/32_dung_tu_dap_an_tay.py --nen firstdance_sotuyen1.zip

VÌ SAO ĐỔI SỐ KEYFRAME CHỨ KHÔNG DÙNG THẲNG
============================================

Người soát ghi theo **tên file ảnh** (`095.jpg` -> `kf_n = 95`) vì đó là thứ họ
nhìn thấy khi mở thư mục. Nhưng BTC chấm theo **`frame_idx`**, hai con số hoàn
toàn khác nhau: `L30_V046` kf95 có `frame_idx = 6613`. Nộp nhầm số keyframe là
sai toàn bộ.

⚠️ Lấy `frame_idx` từ cột của bảng cái, **tuyệt đối không tính lại từ
`pts_time`** — làm tròn lệch 1 frame (bẫy đầu tiên trong CLAUDE.md).

CÁCH GHI ĐÁP ÁN
===============

    "query-p1-1-kis":  ("L30_V046", [95]),
    "query-p1-13-kis": ("L29_V021", range(80, 85)),      # "080 -> 084"
    "query-p1-17-qa":  ("L22_V008", [63, 62, 59], "Tà Pứa"),

Gói Q&A có phần tử thứ ba là **chuỗi đáp án**. Thứ tự khung trong danh sách
chính là thứ hạng nộp — khung chắc nhất để đầu, vì R@1 chiếm 1/5 tổng điểm.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from nop_bai import dong_goi, ghi_goi, soat_zip        # noqa: E402
from schema import AnswerKIS, AnswerQA, AnswerTRAKE    # noqa: E402

# ---------------------------------------------------------------------------
# ĐÁP ÁN NHÓM ĐÃ SOÁT TAY — sửa ở đây cho các đợt sau.
#   (video_id, [số keyframe theo tên file])                   với KIS/TRAKE
#   (video_id, [số keyframe], "chuỗi đáp án")                 với Q&A
# ---------------------------------------------------------------------------
DAP_AN = {
    "query-p1-1-kis":  ("L30_V046", [95]),
    "query-p1-2-kis":  ("L28_V022", [63, 142]),
    # ⚠️ p1-3-qa: nhóm CHƯA có đáp án. Câu hỏi đòi đọc con số trên mặt cân —
    # thuần thị giác, không có chữ nào để bám nên kênh OCR/objects bó tay
    # (xem A26). Điền tạm để gói không rỗng; PHẢI thay trước khi nộp thật.
    "query-p1-3-qa":   ("L27_V009", [266, 254], "không rõ"),
    "query-p1-4-kis":  ("L22_V021", [183, 186, 187]),
    "query-p1-5-kis":  ("L26_V035", [96, 107]),
    "query-p1-6-kis":  ("L22_V023", [10]),
    "query-p1-7-kis":  ("L26_V041", [73, 11]),
    "query-p1-8-kis":  ("L26_V171", [138, 139, 141, 145, 146]),
    "query-p1-9-qa":   ("L21_V003", list(range(252, 261)), "2,15"),
    "query-p1-10-kis": ("L29_V013", [236, 237]),
    "query-p1-11-kis": ("L23_V021", [136]),
    "query-p1-12-kis": ("L22_V008", [77]),
    "query-p1-13-kis": ("L29_V021", list(range(80, 85))),
    "query-p1-14-kis": ("L26_V171", [138, 139, 141, 145, 146]),
    "query-p1-15-qa":  ("L21_V006", list(range(13, 19)), "46"),
    "query-p1-16-trake": ("L24_V031", [1, 14, 25]),
    "query-p1-17-qa":  ("L22_V008", [63, 62, 59], "Tà Pứa"),
    "query-p1-18-kis": ("L26_V389", list(range(132, 141))),
    "query-p1-19-kis": ("L24_V035", list(range(138, 143))),
    "query-p1-20-kis": ("L21_V026", list(range(64, 67))),
    "query-p1-21-kis": ("L22_V011", list(range(135, 138))),
    "query-p1-22-kis": ("L25_V041", list(range(173, 182))),
    "query-p1-23-kis": ("L25_V060", list(range(263, 268))),
    "query-p1-24-kis": ("L29_V001", [226, 227, 228, 229]),
    "query-p1-25-kis": ("L30_V003", [102, 103]),
}


def frame_idx_cua(master, video_id: str, kf: int):
    """`kf_n` -> `frame_idx`. `None` nếu không có, kèm cảnh báo ở chỗ gọi."""
    r = master[(master.video_id == video_id) & (master.kf_n == int(kf))]
    return None if r.empty else int(r.frame_idx.iloc[0])


def main():
    ap = argparse.ArgumentParser(description="dung bai nop tu dap an soat tay")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--ra", default=Path("submission_sotuyen1"), type=Path)
    ap.add_argument("--nen", metavar="FILE.zip")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    goi, so_su_kien, thieu = {}, {}, []

    for ten, muc in sorted(DAP_AN.items()):
        vid, kfs = muc[0], list(muc[1])
        dap = muc[2] if len(muc) > 2 else None
        loai = ten.rsplit("-", 1)[-1]

        frames = []
        for k in kfs:
            f = frame_idx_cua(master, vid, k)
            if f is None:
                thieu.append(f"{ten}: {vid} kf{k} không có trong bảng cái")
            else:
                frames.append(f)
        if not frames:
            thieu.append(f"{ten}: KHÔNG có khung nào hợp lệ")
            continue

        if loai == "trake":
            # Một dòng = một bộ N khung, phải tăng dần theo thời gian.
            goi[ten] = [AnswerTRAKE(vid, sorted(frames))]
            so_su_kien[ten] = len(frames)
        elif loai == "qa":
            goi[ten] = [AnswerQA(vid, f, dap) for f in frames]
        else:
            goi[ten] = [AnswerKIS(vid, f) for f in frames]

        print(f"  {ten:<20} {vid:<10} {len(goi[ten]):>2} dòng  "
              f"kf {kfs[0]}..{kfs[-1]} -> frame {frames[0]}..{frames[-1]}"
              f"{'  | answer ' + repr(dap) if dap else ''}")

    if thieu:
        print("\n⚠️  CẢNH BÁO:")
        for x in thieu:
            print("   ", x)

    d = ghi_goi(goi, a.ra, so_su_kien)     # tự soát; có lỗi thì KHÔNG ghi gì
    print(f"\n✅ Đã ghi {d}  ({len(goi)}/25 gói)")

    if a.nen:
        z = dong_goi(d, a.nen)
        loi, canh = soat_zip(z)
        for x in canh:
            print("⚠️ ", x)
        if loi:
            print(f"\n❌ {len(loi)} LỖI — ĐỪNG NỘP:")
            for x in loi:
                print("   ", x)
            raise SystemExit(1)
        print(f"✅ Đã nén -> {z} — đạt checklist BTC")


if __name__ == "__main__":
    main()
