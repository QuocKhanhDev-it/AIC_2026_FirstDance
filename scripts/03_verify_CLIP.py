"""
03_verify_clip.py — Kiểm chứng phần CÒN LẠI của bảng cái.

02_verify.py chỉ chạy được trên video có sẵn ảnh keyframe (hiện mới có L21).
Script này kiểm chứng liên kết  CSV  <->  clip.npy  cho BẤT KỲ video nào chỉ
cần có file .mp4, không cần ảnh keyframe.

Nguyên lý: trích frame tại pts_time bằng ffmpeg, encode lại bằng đúng CLIP
ViT-B/32, so cosine với vector đã lưu trong clip.npy tại row_id đó.

    cosine >= 0.98  -> KHỚP (chênh lệch chỉ do JPEG + fp16)
    cosine 0.90-0.98-> NGHI NGỜ
    cosine <  0.90  -> LỆCH

Cài thêm:
    pip install torch torchvision open_clip_torch
    (CPU chạy được, ~1-2 giây/ảnh; có GPU thì nhanh hơn nhiều)

Chạy:
    python 03_verify_clip.py --out ./index --n 40
    python 03_verify_clip.py --out ./index --n 40 --group L26   # chỉ 1 nhóm L
"""

import argparse, subprocess, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

MATCH, SUSPECT = 0.98, 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("./index"), type=Path)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--group", default=None, help="lọc theo nhóm L, vd L26")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    m = pd.read_parquet(a.out / "master.parquet")
    clip_mat = np.load(a.out / "clip.npy", mmap_mode="r")
    assert len(m) == clip_mat.shape[0], "master và clip.npy lệch số dòng!"

    m = m[m.video_path.notna() & m.has_clip]
    if a.group:
        m = m[m.video_id.str.startswith(a.group)]
    if m.empty:
        raise SystemExit("Không có dòng nào đủ điều kiện (cần video_path). "
                         "Tải thêm file .mp4 rồi chạy lại 00_discover.py.")

    rng = np.random.default_rng(a.seed)
    vids = m.video_id.unique()
    pick = rng.choice(vids, size=min(a.n, len(vids)), replace=False)
    # groupby.sample: pandas 3.0 loại cột nhóm khỏi groupby.apply, dùng cách này
    # vừa gọn vừa giữ nguyên mọi cột.
    sample = (m[m.video_id.isin(pick)]
              .groupby("video_id", as_index=False)
              .sample(n=1, random_state=a.seed)
              .head(a.n))

    print("Nạp CLIP ViT-B/32 ...")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    model = model.to(dev).eval()

    print(f"Kiểm tra {len(sample)} keyframe trên {sample.video_id.nunique()} video "
          f"(thiết bị: {dev})\n")

    res = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "f.png"
        for k, r in enumerate(sample.itertuples(index=False), 1):
            rc = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{r.pts_time:.3f}",
                 "-i", r.video_path, "-frames:v", "1", str(tmp), "-y"],
                capture_output=True)
            if rc.returncode != 0 or not tmp.exists():
                res.append({"video_id": r.video_id, "row_id": r.row_id,
                            "cosine": np.nan, "ket_luan": "khong_trich_duoc"})
                continue

            img = preprocess(Image.open(tmp)).unsqueeze(0).to(dev)
            with torch.no_grad():
                v = model.encode_image(img)[0].cpu().numpy().astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9

            stored = np.asarray(clip_mat[r.row_id], dtype=np.float32)
            cos = float(v @ stored)

            kl = ("KHOP" if cos >= MATCH else
                  "NGHI_NGO" if cos >= SUSPECT else "LECH")
            res.append({"video_id": r.video_id, "row_id": r.row_id,
                        "kf_n": r.kf_n, "frame_idx": r.frame_idx,
                        "pts_time": round(r.pts_time, 2), "fps": r.fps,
                        "cosine": round(cos, 4), "ket_luan": kl})
            print(f"  [{k}/{len(sample)}] {r.video_id} kf_n={r.kf_n} "
                  f"cos={cos:.4f}  {kl}")

    out = pd.DataFrame(res)
    name = f"verify_clip{'_'+a.group if a.group else ''}.csv"
    out.to_csv(a.out / name, index=False)

    print("\n" + "=" * 60)
    vc = out.ket_luan.value_counts()
    for k, v in vc.items():
        print(f"  {k:<20} {v:>4}  ({v/len(out)*100:.0f}%)")
    ok = vc.get("KHOP", 0) / len(out)
    print()
    if ok >= 0.95:
        print("  ✅ Liên kết CSV <-> clip.npy ĐÚNG cho nhóm đã kiểm tra.")
    else:
        print("  ❌ LỆCH. Vector trong clip.npy không ứng với dòng CSV cùng vị trí.")
        print("     Kiểm tra riêng các video có fps lạ (26.44, 29.97).")
    print(f"\n  Chi tiết: index/{name}")
    print("=" * 60)


if __name__ == "__main__":
    main()