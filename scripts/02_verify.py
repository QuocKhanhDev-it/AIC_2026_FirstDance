"""
02_verify.py - Bước 3/3 của Giai đoạn 0. QUAN TRỌNG NHẤT.

Chứng minh bằng máy rằng: ảnh keyframe ở dòng thứ i THẬT SỰ là frame số
`frame_idx` trong video. Nếu sai, mọi module xây trên bảng cái đều sai và
không ai phát hiện được cho tới tuần 4.

Cách làm: lấy ngẫu nhiên N dòng, dùng ffmpeg trích frame tại `pts_time`,
so sánh với ảnh keyframe bằng tương quan chuẩn hóa trên ảnh xám 64x64.

    corr >= 0.95  -> KHỚP
    corr 0.7-0.95 -> NGHI NGỜ (cùng cảnh, khác frame)
    corr <  0.7   -> KHÔNG KHỚP

Script cũng thử các lệch chuẩn (dòng i-1, i+1, và các cách đánh số khác)
để chỉ ra chính xác đang sai kiểu gì.

Chạy:
    python 02_verify.py --out ./index --n 60
    python 02_verify.py --out ./index --n 498 --group L26   # kiểm HẾT một nhóm

Yêu cầu: ffmpeg trong PATH, pip install pillow
"""

import argparse, subprocess, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MATCH, SUSPECT = 0.95, 0.70
# Biên độ tối thiểu so với dòng kề để vẫn kết luận ghép đúng dù corr thấp.
# Hiệu chuẩn trên L22: các mẫu corr 0,67-0,91 đều vượt dòng kề >= 0,37.
BIEN_DO = 0.30


def sig(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((64, 64)), dtype=float)
    return (a - a.mean()) / (a.std() + 1e-6)


def frame_at(video: str, t: float, tmp: Path) -> np.ndarray | None:
    out = tmp / "f.png"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video,
         "-frames:v", "1", str(out), "-y"],
        capture_output=True)
    if r.returncode != 0 or not out.exists():
        return None
    try:
        return sig(Image.open(out))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("./index"), type=Path)
    ap.add_argument("--n", type=int, default=60, help="số mẫu kiểm tra")
    ap.add_argument("--group", default=None,
                    help="lọc theo nhóm L, vd L26. Một máy giữ nhiều nhóm mà "
                         "không lọc thì mẫu bị chia đều, nhóm đông nuốt hết chỗ.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    m = pd.read_parquet(a.out / "master.parquet")
    m = m[m.kf_path.notna() & m.video_path.notna()]
    if a.group:
        m = m[m.video_id.str.startswith(a.group)]
    if m.empty:
        raise SystemExit(
            f"Không có dòng nào đủ kf_path + video_path"
            f"{' cho nhóm ' + a.group if a.group else ''}.\n"
            "Script này cần CẢ ảnh keyframe LẪN file .mp4. Nếu chỉ có .mp4 thì "
            "dùng 03_verify_CLIP.py.")

    # lấy mẫu trải đều trên nhiều video
    rng = np.random.default_rng(a.seed)
    vids = m.video_id.unique()
    pick_v = rng.choice(vids, size=min(a.n, len(vids)), replace=False)
    # groupby.sample: pandas 3.0 loại cột nhóm khỏi groupby.apply, dùng cách này
    # vừa gọn vừa giữ nguyên mọi cột.
    sample = (m[m.video_id.isin(pick_v)]
              .groupby("video_id", as_index=False)
              .sample(n=1, random_state=a.seed)
              .head(a.n))

    print(f"Kiểm tra {len(sample)} keyframe trên {sample.video_id.nunique()} video...\n")
    res = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for k, r in enumerate(sample.itertuples(index=False), 1):
            try:
                ref = sig(Image.open(r.kf_path))
            except Exception as e:
                res.append({"video_id": r.video_id, "kf_name": r.kf_name,
                            "corr": np.nan, "ket_luan": f"loi_doc_anh:{e}"})
                continue

            cands = {"dung_dong": r.pts_time}
            same = m[m.video_id == r.video_id].sort_values("kf_n")
            arr = same.reset_index(drop=True)
            j = arr.index[arr.kf_n == r.kf_n]
            if len(j):
                j = int(j[0])
                if j > 0:
                    cands["lech_-1"] = float(arr.pts_time.iloc[j - 1])
                if j + 1 < len(arr):
                    cands["lech_+1"] = float(arr.pts_time.iloc[j + 1])

            scores = {}
            for name, t in cands.items():
                s = frame_at(r.video_path, t, tmp)
                scores[name] = float((ref * s).mean()) if s is not None else np.nan

            best = max((v, k2) for k2, v in scores.items() if not np.isnan(v))
            c = scores.get("dung_dong", np.nan)
            # Biên độ so với dòng kề: đây mới là dấu hiệu ghép ĐÚNG hay LỆCH.
            # Tương quan pixel tuyệt đối sụt mạnh trên cảnh động (đồ họa chuyển
            # cảnh, hoạt hình đài) dù ghép hoàn toàn đúng — đo thật trên
            # L22_V013/116.jpg: corr chỉ 0,675 nhưng đồng hồ trên hình đọc
            # cùng 18:36:43 với frame trích từ video, tức đúng giây.
            ke = [v for k2, v in scores.items() if k2 != "dung_dong" and not np.isnan(v)]
            bien_do = c - max(ke) if ke and not np.isnan(c) else np.nan

            if np.isnan(c):
                kl = "khong_trich_duoc_frame"
            elif c >= MATCH:
                kl = "KHOP"
            elif best[1] != "dung_dong" and best[0] >= MATCH:
                kl = f"LECH_INDEX ({best[1]})"
            elif not np.isnan(bien_do) and bien_do >= BIEN_DO:
                # thấp hơn ngưỡng nhưng vượt xa dòng kề -> vẫn là ghép đúng
                kl = "KHOP_YEU"
            elif c >= SUSPECT:
                kl = "NGHI_NGO"
            else:
                kl = "KHONG_KHOP"

            res.append({"video_id": r.video_id, "kf_name": r.kf_name,
                        "frame_idx": r.frame_idx, "pts_time": round(r.pts_time, 2),
                        "corr": round(c, 4) if not np.isnan(c) else np.nan,
                        "bien_do": round(bien_do, 4) if not np.isnan(bien_do) else np.nan,
                        **{f"corr_{k2}": round(v, 4) for k2, v in scores.items()
                           if k2 != "dung_dong"},
                        "ket_luan": kl})
            print(f"  [{k}/{len(sample)}] {r.video_id} {r.kf_name}  "
                  f"corr={c:.3f}  {kl}")

    out = pd.DataFrame(res)
    ten = f"verify_report{'_' + a.group if a.group else ''}.csv"
    out.to_csv(a.out / ten, index=False)

    print("\n" + "=" * 64)
    print("KẾT QUẢ")
    vc = out.ket_luan.value_counts()
    for k, v in vc.items():
        print(f"  {k:<28} {v:>4}  ({v/len(out)*100:.0f}%)")
    ok = (vc.get("KHOP", 0) + vc.get("KHOP_YEU", 0)) / len(out)
    if vc.get("KHOP_YEU", 0):
        print(f"\n  ({vc['KHOP_YEU']} mẫu KHOP_YEU: tương quan thấp nhưng vượt xa dòng kề")
        print("   -> ghép đúng, chỉ là cảnh động/chuyển cảnh. Tính là đạt.)")
    print("")
    if ok >= 0.95:
        print("  -> BẢNG CÁI ĐÚNG. Được phép sang Giai đoạn 1.")
    elif any(str(k).startswith("LECH_INDEX") for k in vc.index):
        print("  -> LỆCH CHỈ SỐ. Ảnh keyframe không ứng với dòng CSV cùng vị trí.")
        print("     Sửa cách ghép trong 01_build_index.py rồi chạy lại.")
    else:
        print("  -> KHÔNG KHỚP. Có thể ảnh keyframe thuộc video khác, hoặc")
        print(f"     frame_idx tính theo fps khác. Xem index/{ten}.")
    print(f"\n  Chi tiết: index/{ten}")
    print("=" * 64)


if __name__ == "__main__":
    main()
