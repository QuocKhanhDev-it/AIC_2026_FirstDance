"""
06_tim.py — Gõ một câu, ra danh sách khoảnh khắc kèm phút:giây.

Đây là kênh CLIP ở dạng dùng được ngay, chưa phải bản hoàn chỉnh cho bài thi
(chưa có RRF, chưa ghép BM25/objects) nhưng đủ để tìm cảnh trong kho và để
Khánh soạn tập dev.

Chạy:
    python scripts/06_tim.py "a person riding a motorbike on the street"
    python scripts/06_tim.py "boats on a river" --k 20
    python scripts/06_tim.py "news anchor" --video L22_V008      # trong 1 video
    python scripts/06_tim.py "cooking" --co-anh                  # chỉ nơi đã tải ảnh

LƯU Ý:
  - Truy vấn nên viết TIẾNG ANH. CLIP gốc yếu với tiếng Việt.
  - Cột `frame_idx` là giá trị NỘP CHO BTC. Đừng tự tính từ pts_time —
    làm tròn lệch 1 frame (đo thật: 1061,36 × 25fps = 26534, thật là 26533).
  - Điểm cosine của CLIP ViT-B/32 thường chỉ 0,25-0,40. Model LUÔN trả về
    top-1 kể cả khi kho KHÔNG có cảnh đó. Đo thật: truy vấn "people playing
    football on a field" trả về một video ôn thi môn Toán với cos = 0,311 —
    sai hoàn toàn nhưng điểm không hề thấp bất thường.

    => KHÔNG CÓ ngưỡng cosine nào đáng tin để tự động loại kết quả sai.
    Ngưỡng 0,30 dưới đây chỉ là mốc thô, PHẢI hiệu chuẩn lại trên tập dev
    của Khánh. Đây chính là lý do bốn kênh phải hợp nhất bằng RRF thay vì
    tin một mình CLIP.

Yêu cầu: pip install -r requirements-clip.txt
"""

import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_TAG = "ViT-B-32-quickgelu"    # KHÔNG đổi — xem kế hoạch v4 mục A6


def hms(giay: float) -> str:
    return f"{int(giay // 60)}:{giay % 60:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="truy vấn, nên viết tiếng Anh")
    ap.add_argument("--index", default=Path("./index"), type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--video", default=None, help="giới hạn trong một video_id")
    ap.add_argument("--co-anh", action="store_true",
                    help="chỉ những keyframe đã tải ảnh về máy này")
    ap.add_argument("--moi-video", type=int, default=2,
                    help="tối đa bao nhiêu kết quả mỗi video (ràng buộc đa dạng "
                         "của PHẦN C: mỗi video <= 2 slot trong top-5)")
    a = ap.parse_args()

    import torch, open_clip

    m = pd.read_parquet(a.index / "master.parquet")
    clip = np.load(a.index / "clip.npy", mmap_mode="r")
    assert len(m) == clip.shape[0], "master.parquet và clip.npy lệch số dòng!"

    model, _, _ = open_clip.create_model_and_transforms(MODEL_TAG, pretrained="openai")
    tok = open_clip.get_tokenizer(MODEL_TAG)
    model.eval()
    with torch.no_grad():
        v = model.encode_text(tok([a.query]))[0].numpy().astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9

    sim = np.asarray(clip) @ v
    if a.video:
        sim[(m.video_id != a.video).values] = -9
    if a.co_anh:
        sim[m.kf_path.isna().values] = -9

    # lấy dư rồi áp ràng buộc đa dạng: sai một video là mất cả chùm kết quả
    thu = m.iloc[np.argsort(-sim)[: a.k * a.moi_video + 50]].assign(
        cos=np.sort(sim)[::-1][: a.k * a.moi_video + 50])
    thu = thu[thu.cos > -1]
    ket = thu.groupby("video_id", group_keys=False).head(a.moi_video) \
             .sort_values("cos", ascending=False).head(a.k)

    print(f'"{a.query}"   →  {len(ket)} khoảnh khắc\n')
    print(f"{'cos':>6}  {'video_id':<10} {'thời điểm':>9}  {'frame_idx':>9}  {'fps':>5}  ảnh  tiêu đề")
    print("-" * 108)
    for r in ket.itertuples():
        print(f"{r.cos:6.3f}  {r.video_id:<10} {hms(r.pts_time):>9}  {r.frame_idx:>9}  "
              f"{r.fps:>5}  {'có' if isinstance(r.kf_path, str) else ' - ':<3}  {r.title[:44]}")

    tot = ket.iloc[0]
    print()
    if tot.cos < 0.32:
        print(f"  ⚠️  Điểm cao nhất chỉ {tot.cos:.3f}. Mốc thô, CHƯA hiệu chuẩn —")
        print("      kết quả sai vẫn có thể đạt 0,31. Kiểm bằng mắt trước khi tin.")
    if isinstance(tot.video_path, str):
        print("  Xem tận mắt khoảnh khắc mạnh nhất:")
        print(f'    ffmpeg -ss {tot.pts_time:.3f} -i "{tot.video_path}" -frames:v 1 xem.png -y')
    else:
        print(f"  (Chưa tải video của {tot.video_id} nên không trích frame ra xem được.)")


if __name__ == "__main__":
    main()
