"""
10_contact_sheet.py — Dựng lưới ảnh để DUYỆT BẰNG MẮT khi soạn tập dev.

Vì sao cần: đáp án của tập dev phải tìm bằng mắt, KHÔNG được tìm bằng chính
`06_tim.py`. Tập dev tạo bằng công cụ đang được đem ra chấm chỉ chứa những
khoảnh khắc mà CLIP vốn đã tìm được — đo trên đó thì CLIP luôn trông tốt và
mọi cải tiến đều trông vô dụng.

Mỗi ô có ghi `row_id` ngay trên ảnh. Đó là con số duy nhất cần chép lại: từ
`row_id` suy ra được video_id, frame_idx, pts_time, fps — ngược lại thì không.

Một video ~203 keyframe. Lấy thưa 1/10 còn ~20 ô, đủ để thấy video nói về gì
mà file chỉ vài trăm KB. Chọn được khoảnh khắc rồi mới dựng lại dày quanh đó.

    # duyệt nhanh cả nhóm L, mỗi video một file
    python scripts/10_contact_sheet.py --nhom L21 --thua 10

    # đã ưng một video -> xem dày quanh vùng quan tâm
    python scripts/10_contact_sheet.py --video L21_V003 --thua 1 --tu 120 --den 180

CHIA SẺ GIỮA CÁC MÁY: ảnh keyframe cả kho là 30,5 GB, không ai gửi nổi. Contact
sheet thưa 1/10 cho toàn bộ 873 video chỉ khoảng 350 MB — đẩy Drive được. Mỗi
máy dựng cho nhóm L mình giữ rồi đẩy lên, cả nhóm cùng duyệt được toàn kho.
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

GOC = Path(__file__).resolve().parent.parent
RONG, CAO, COT = 320, 180, 5
NHAN = 22          # dải chữ dưới mỗi ô


def _font():
    for ten in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(ten, 13)
        except OSError:
            continue
    return ImageFont.load_default()


def dung_sheet(lat: pd.DataFrame, ra: Path, tieu_de: str) -> Path | None:
    """Một file ảnh cho một video. Trả None nếu không đọc được ảnh nào."""
    font = _font()
    n = len(lat)
    hang = -(-n // COT)
    sheet = Image.new("RGB", (RONG * COT, (CAO + NHAN) * hang + 26), "white")
    ve = ImageDraw.Draw(sheet)
    ve.text((6, 6), tieu_de, fill="black", font=font)

    doc_duoc = 0
    for i, r in enumerate(lat.itertuples()):
        x, y = (i % COT) * RONG, (i // COT) * (CAO + NHAN) + 26
        try:
            im = Image.open(r.kf_path).convert("RGB")
            im.thumbnail((RONG - 4, CAO - 4))
            sheet.paste(im, (x + 2, y + 2))
            doc_duoc += 1
        except Exception:
            ve.rectangle([x + 2, y + 2, x + RONG - 2, y + CAO - 2], fill="#eee")
            ve.text((x + 8, y + CAO // 2), "không đọc được ảnh", fill="red", font=font)
        # row_id là thứ DUY NHẤT cần chép lại khi soạn tập dev
        ve.text((x + 4, y + CAO + 3),
                f"row_id {r.row_id}   kf {r.kf_n}   {r.pts_time:.1f}s",
                fill="#0645ad", font=font)

    if not doc_duoc:
        return None
    ra.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(ra, quality=72, optimize=True)
    return ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--out", default=GOC / "dev" / "sheets", type=Path)
    ap.add_argument("--video", default=None, help="một video_id cụ thể")
    ap.add_argument("--nhom", default=None, help="cả một nhóm L, vd L21")
    ap.add_argument("--thua", type=int, default=10,
                    help="lấy 1 ảnh mỗi N keyframe (mặc định 10)")
    ap.add_argument("--tu", type=int, default=None, help="chỉ từ kf_n này")
    ap.add_argument("--den", type=int, default=None, help="đến kf_n này")
    a = ap.parse_args()

    m = pd.read_parquet(a.index / "master.parquet")
    m = m[m.kf_path.notna()]
    if m.empty:
        raise SystemExit(
            "Máy này chưa tải ảnh keyframe nào. Cần gói Keyframes_* của nhóm L "
            "bạn giữ, hoặc xin contact sheet từ máy khác (~350 MB cho cả kho).")

    if a.video:
        m = m[m.video_id == a.video]
    elif a.nhom:
        m = m[m.video_id.str.startswith(a.nhom)]
    if a.tu is not None:
        m = m[m.kf_n >= a.tu]
    if a.den is not None:
        m = m[m.kf_n <= a.den]
    if m.empty:
        co = sorted(set(pd.read_parquet(a.index / "master.parquet")
                        .dropna(subset=["kf_path"]).video_id.str[:3]))
        raise SystemExit(f"Không có ảnh khớp bộ lọc. Nhóm L có ảnh: {co}")

    lam = 0
    for vid, lat in m.groupby("video_id"):
        lat = lat.iloc[::a.thua]
        f = dung_sheet(lat, a.out / f"{vid}.jpg",
                       f"{vid}   {len(lat)} ô (thưa 1/{a.thua})   "
                       f"{str(lat.title.iloc[0])[:70]}")
        if f:
            lam += 1
            print(f"  {f}   {f.stat().st_size / 1024:.0f} KB   {len(lat)} ô")

    print(f"\n{lam} contact sheet -> {a.out}")
    if lam:
        print("\nDuyệt bằng mắt, thấy khoảnh khắc đáng hỏi thì chép lại `row_id`\n"
              "ghi trên ô đó. Đừng dùng 06_tim.py để tìm — xem docs/07_lam_tap_dev.md")


if __name__ == "__main__":
    main()
