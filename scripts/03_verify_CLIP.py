"""
03_verify_clip.py — Kiểm chứng phần CÒN LẠI của bảng cái.

02_verify.py chỉ chạy được trên video có sẵn ảnh keyframe (hiện mới có L21).
Script này kiểm chứng liên kết  CSV  <->  clip.npy  cho BẤT KỲ video nào chỉ
cần có file .mp4, không cần ảnh keyframe.

Nguyên lý: trích frame tại pts_time bằng ffmpeg, encode lại bằng đúng CLIP
ViT-B/32, so cosine với vector đã lưu trong clip.npy tại row_id đó.

Hai phép thử, phép thứ hai mới là phép quyết định:

1. COSINE TUYỆT ĐỐI — nhạy với khác biệt tiền xử lý (JPEG vs PNG, cách
   resize, phiên bản model). Đo trên L21 đã biết chắc đúng: 0.973-0.996.

       cosine >= 0.95  -> đạt
       cosine 0.85-0.95-> nghi ngờ
       cosine <  0.85  -> lệch

2. THỨ HẠNG TRONG VIDEO — vector vừa encode phải giống vector đã lưu tại
   row_id đó HƠN mọi keyframe khác của cùng video. Phép này không phụ thuộc
   ngưỡng nên đáng tin hơn hẳn. Đo trên L21: 8/8 đúng hạng 1, cách ứng viên
   nhì trung bình 0.12.

Kết luận KHỚP đòi hỏi ĐỦ CẢ HAI.

LƯU Ý: phải dùng tag "ViT-B-32-quickgelu". Model OpenAI CLIP dùng QuickGELU;
nạp "ViT-B-32" thường làm cosine tụt ~0.04 (0.989 -> 0.951) và mọi mẫu trượt
ngưỡng dù dữ liệu hoàn toàn đúng.

Cài thêm:
    pip install torch torchvision open_clip_torch
    (CPU chạy được, ~1-2 giây/ảnh; có GPU thì nhanh hơn nhiều)

Chạy:
    python 03_verify_clip.py --out ./index --n 40
    python 03_verify_clip.py --out ./index --n 40 --group L26   # chỉ 1 nhóm L
"""

import argparse, subprocess, tempfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

warnings.filterwarnings("ignore")

# Hiệu chuẩn trên L21 — nhóm đã được 02_verify.py chứng minh là ghép đúng.
MATCH, SUSPECT = 0.95, 0.85
MODEL_TAG = "ViT-B-32-quickgelu"     # KHÔNG đổi thành "ViT-B-32", xem docstring


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

    print(f"Nạp CLIP {MODEL_TAG} ...")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_TAG, pretrained="openai")
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

            cos = float(v @ np.asarray(clip_mat[r.row_id], dtype=np.float32))

            # Phép thử quyết định: so với TOÀN BỘ keyframe của cùng video.
            # Vector vừa encode phải giống dòng row_id của nó nhất. Phép này
            # không phụ thuộc ngưỡng nên bắt được lệch chỉ số mà cosine
            # tuyệt đối có thể bỏ qua.
            ids = m.loc[m.video_id == r.video_id, "row_id"].to_numpy()
            sims = np.asarray(clip_mat[ids], dtype=np.float32) @ v
            order = np.argsort(-sims)
            hang = int(np.where(ids[order] == r.row_id)[0][0]) + 1
            cach_biet = cos - float(sims[order[1 if hang == 1 else 0]])

            if hang != 1:
                kl = f"LECH_INDEX (dung hang {hang}/{len(ids)})"
            elif cos >= MATCH:
                kl = "KHOP"
            elif cos >= SUSPECT:
                kl = "NGHI_NGO"
            else:
                kl = "LECH"

            res.append({"video_id": r.video_id, "row_id": r.row_id,
                        "kf_n": r.kf_n, "frame_idx": r.frame_idx,
                        "pts_time": round(r.pts_time, 2), "fps": r.fps,
                        "cosine": round(cos, 4), "hang": hang,
                        "so_keyframe": len(ids),
                        "cach_biet": round(cach_biet, 4), "ket_luan": kl})
            print(f"  [{k}/{len(sample)}] {r.video_id} kf_n={r.kf_n} "
                  f"cos={cos:.4f} hang={hang}/{len(ids)} "
                  f"cach_biet={cach_biet:+.4f}  {kl}")

    out = pd.DataFrame(res)
    name = f"verify_clip{'_'+a.group if a.group else ''}.csv"
    out.to_csv(a.out / name, index=False)

    print("\n" + "=" * 60)
    vc = out.ket_luan.value_counts()
    for k, v in vc.items():
        print(f"  {k:<30} {v:>4}  ({v/len(out)*100:.0f}%)")
    ok = vc.get("KHOP", 0) / len(out)

    if "hang" in out:
        dung_hang = (out.hang == 1).sum()
        print()
        print(f"  đúng hạng 1        {dung_hang}/{len(out)}")
        print(f"  cosine             trung bình {out.cosine.mean():.4f}, "
              f"thấp nhất {out.cosine.min():.4f}")
        print(f"  cách biệt hạng 2   trung bình {out.cach_biet.mean():+.4f}")

    print()
    if ok >= 0.95:
        print("  ✅ Liên kết CSV <-> clip.npy ĐÚNG cho nhóm đã kiểm tra.")
    elif any(str(k).startswith("LECH_INDEX") for k in vc.index):
        print("  ❌ LỆCH CHỈ SỐ. Vector khớp với dòng KHÁC trong cùng video.")
        print("     Xem cột `hang` để biết lệch mấy dòng.")
    else:
        print("  ⚠️  Đúng hạng 1 nhưng cosine thấp — nhiều khả năng do tiền xử lý")
        print("     (model tag, cách resize, JPEG) chứ không phải lệch chỉ số.")
        print("     Kiểm tra riêng các video có fps lạ (26.44, 29.97).")
    print(f"\n  Chi tiết: index/{name}")
    print("=" * 60)


if __name__ == "__main__":
    main()