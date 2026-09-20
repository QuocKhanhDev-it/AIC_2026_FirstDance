"""
00_discover.py - Bước 1/3 của Giai đoạn 0.

Quét các thư mục tải về (dù nằm rải rác, tên khác nhau, lồng nhau bao nhiêu
tầng) và lập bản đồ: mỗi video_id có những tài sản gì, nằm ở đâu.

KHÔNG di chuyển, KHÔNG copy file nào. Chỉ ghi lại đường dẫn.

Chạy:
    python 00_discover.py --roots "D:/Downloads" --out ./index

Có thể truyền nhiều root:
    python 00_discover.py --roots "D:/Downloads" "E:/AIC_data" --out ./index

Output: index/paths.parquet  +  index/discover_report.txt
"""

import argparse, re, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# L21_V030, L26_V383, ...
VIDEO_ID = re.compile(r"^L\d{2}_V\d{3,4}$")
IMG_EXT = {".jpg", ".jpeg", ".png"}

# thư mục bỏ qua cho nhanh
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information", "node_modules",
             ".git", "__pycache__", "venv", ".venv", "cache"}


def scan(roots: list[Path]) -> dict[str, dict]:
    assets: dict[str, dict] = defaultdict(dict)
    n_seen = 0

    for root in roots:
        if not root.is_dir():
            print(f"  ! bỏ qua (không phải thư mục): {root}")
            continue
        print(f"  quét {root} ...")
        for p in root.rglob("*"):
            n_seen += 1
            if n_seen % 200_000 == 0:
                print(f"    ... đã duyệt {n_seen:,} mục")
            if any(part in SKIP_DIRS for part in p.parts):
                continue

            # ---- thư mục tên đúng dạng video_id -> keyframes hoặc objects ----
            if p.is_dir() and VIDEO_ID.match(p.name):
                vid = p.name
                has_img = any(c.suffix.lower() in IMG_EXT for c in p.iterdir())
                has_json = any(c.suffix.lower() == ".json" for c in p.iterdir())
                if has_img:
                    assets[vid]["keyframe_dir"] = str(p)
                if has_json and not has_img:
                    assets[vid]["object_dir"] = str(p)
                continue

            if not p.is_file():
                continue

            stem, ext = p.stem, p.suffix.lower()
            if not VIDEO_ID.match(stem):
                continue

            if ext == ".mp4":
                assets[stem]["video"] = str(p)
            elif ext == ".csv":
                assets[stem]["csv"] = str(p)
            elif ext == ".npy":
                assets[stem]["clip"] = str(p)
            elif ext == ".json":
                # json NẰM TRONG thư mục tên video_id  -> object của keyframe
                # json NẰM NGOÀI (tên file = video_id) -> metadata video
                if not VIDEO_ID.match(p.parent.name):
                    assets[stem]["media"] = str(p)

    print(f"  duyệt xong {n_seen:,} mục.\n")
    return assets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, type=Path)
    ap.add_argument("--out", default=Path("./index"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    print("QUÉT DỮ LIỆU")
    assets = scan(a.roots)
    if not assets:
        sys.exit("Không tìm thấy video_id nào dạng Lxx_Vyyy. Kiểm tra lại --roots.")

    cols = ["video", "keyframe_dir", "csv", "object_dir", "media", "clip"]
    df = pd.DataFrame([{"video_id": v, **{c: assets[v].get(c) for c in cols}}
                       for v in sorted(assets)])
    df.to_parquet(a.out / "paths.parquet", index=False)

    # ---------------- BÁO CÁO ----------------
    lines = []
    add = lines.append
    add("=" * 64)
    add(f"Tìm thấy {len(df):,} video_id")
    add("")
    add("ĐỘ PHỦ TỪNG LOẠI TÀI SẢN")
    for c in cols:
        n = df[c].notna().sum()
        add(f"  {c:<14} {n:>6,} / {len(df):,}   ({n/len(df)*100:5.1f}%)")

    add("")
    add("PHÂN BỐ THEO NHÓM L")
    df["Lgroup"] = df.video_id.str[:3]
    g = df.groupby("Lgroup").agg(
        so_video=("video_id", "size"),
        co_video=("video", lambda s: s.notna().sum()),
        co_csv=("csv", lambda s: s.notna().sum()),
        co_clip=("clip", lambda s: s.notna().sum()),
        co_keyframe=("keyframe_dir", lambda s: s.notna().sum()),
        co_object=("object_dir", lambda s: s.notna().sum()),
        co_media=("media", lambda s: s.notna().sum()),
    )
    add(g.to_string())

    # video thiếu thứ quan trọng
    critical = ["csv", "clip", "keyframe_dir"]
    bad = df[df[critical].isna().any(axis=1)]
    add("")
    if len(bad):
        add(f"!! {len(bad):,} video THIẾU tài sản then chốt (csv/clip/keyframe).")
        add("   Xem index/thieu_tai_san.csv")
        bad[["video_id"] + cols].to_csv(a.out / "thieu_tai_san.csv", index=False)
    else:
        add("Mọi video đều đủ csv + clip + keyframe.")

    add("")
    add("GỢI Ý BƯỚC TIẾP: python 01_build_index.py --out ./index")
    add("=" * 64)

    report = "\n".join(lines)
    print(report)
    (a.out / "discover_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
