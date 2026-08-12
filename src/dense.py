"""
dense.py — Kênh 1: truy hồi bằng vector ảnh (CLIP / SigLIP2 / bất kỳ).

Bóc lõi từ `scripts/06_tim.py` thành module gọi được, để Khánh chấm tập dev
bằng code thay vì đọc màn hình.

KHÔNG DÍNH VÀO MỘT MODEL NÀO. Số chiều đọc từ chính ma trận, tên model đọc từ
file `.json` cạnh nó (nếu có). Nhờ vậy khi máy GPU sinh ra `clip_siglip2.npy`
1152 chiều thì chỉ đổi tham số `--matrix`, không sửa dòng code nào:

    kenh = KenhAnh('./index')                              # ViT-B/32 của BTC
    kenh = KenhAnh('./index', matrix='clip_siglip2.npy')   # SigLIP2

Dùng:
    from dense import KenhAnh
    kenh = KenhAnh('./index')          # nạp model + ma trận MỘT lần
    kq = kenh.tim("a person riding a motorbike", k=100)

Chạy trực tiếp để lấy mốc cho bài test cố định:
    python src/dense.py --ghi-moc

⚠️ KHÔNG có ngưỡng cosine nào đáng tin để tự loại kết quả sai. Đo thật: truy
vấn "people playing football on a field" trả về một video ôn thi môn Toán với
cos = 0,311 — sai hoàn toàn nhưng điểm không thấp bất thường. Model LUÔN trả
về top-1 kể cả khi kho không có cảnh đó. Đừng xây logic lọc theo ngưỡng ở đây;
việc phân định để RRF và tập dev lo.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .schema import Candidate
except ImportError:                     # chạy trực tiếp: python src/dense.py
    from schema import Candidate

# Mặc định cho ma trận BTC phát. ĐỪNG đổi thành "ViT-B-32" — bẫy A6: sai biến
# thể làm cosine tụt 0,9913 -> 0,9513 mà KHÔNG ném lỗi nào.
MODEL_MAC_DINH = "ViT-B-32-quickgelu"
PRETRAINED_MAC_DINH = "openai"


class KenhAnh:
    """Giữ model + ma trận trong RAM, dùng lại cho mọi truy vấn.

    Nạp model tốn vài giây. `06_tim.py` nạp lại mỗi lần chạy — chấm 50 câu dev
    kiểu đó là nạp 50 lần. Lớp này nạp một lần.
    """

    def __init__(self, index_dir="./index", matrix="clip.npy",
                 model=None, pretrained=None, mmap=False):
        d = Path(index_dir)
        self.master = pd.read_parquet(d / "master.parquet")
        self.mat = np.load(d / matrix, mmap_mode="r" if mmap else None)

        if len(self.master) != self.mat.shape[0]:
            raise SystemExit(
                f"master.parquet ({len(self.master)} dòng) và {matrix} "
                f"({self.mat.shape[0]} dòng) LỆCH NHAU. Dừng — mọi kết quả sau "
                f"đây sẽ trỏ nhầm keyframe.")

        # Tên model đọc từ .json cạnh ma trận nếu có (do 08_encode.py ghi ra).
        # Không có thì coi như ma trận của BTC.
        canh = d / (Path(matrix).stem + ".json")
        ghi_chu = json.loads(canh.read_text("utf-8")) if canh.exists() else {}
        self.model_tag = model or ghi_chu.get("model", MODEL_MAC_DINH)
        self.pretrained = pretrained or ghi_chu.get("pretrained", PRETRAINED_MAC_DINH)
        self.chieu = self.mat.shape[1]

        import torch, open_clip
        self._torch = torch
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.model_tag, pretrained=self.pretrained)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(self.model_tag)

        # Model và ma trận phải cùng không gian vector. Số chiều lệch là dấu
        # hiệu rõ ràng nhất, và bắt được ngay lúc khởi tạo thay vì để kết quả
        # sai âm thầm.
        thu = self.encode_text("kiểm tra số chiều")
        if thu.shape[0] != self.chieu:
            raise SystemExit(
                f"Model {self.model_tag}/{self.pretrained} sinh vector "
                f"{thu.shape[0]} chiều nhưng {matrix} có {self.chieu} chiều. "
                f"Sai cặp model/ma trận.")

    def encode_text(self, cau: str) -> np.ndarray:
        """Vector truy vấn, đã chuẩn hóa L2 (bắt buộc, để `M @ q` là cosine)."""
        with self._torch.no_grad():
            v = self.model.encode_text(self.tok([cau]))[0].numpy().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def tim(self, cau, k=100, video_id=None, chi_co_anh=False,
            moi_video=None) -> list[Candidate]:
        """Trả về tối đa `k` ứng viên, điểm cao xuống thấp.

        `cau` nhận cả một chuỗi lẫn danh sách chuỗi. Nhiều biến thể của cùng
        một ý thì lấy ĐIỂM CAO NHẤT trên từng keyframe, không lấy trung bình:
        một cách diễn đạt trúng là đủ, không nên bị các cách diễn đạt trượt
        kéo xuống.

        `moi_video` là ràng buộc đa dạng của PHẦN C mục 2. Để None ở tầng kênh
        (mặc định) và áp một lần duy nhất sau khi RRF — áp sớm sẽ cắt mất ứng
        viên mà các kênh khác còn cần.
        """
        cac_cau = [cau] if isinstance(cau, str) else list(cau)
        sim = np.max([np.asarray(self.mat) @ self.encode_text(c)
                      for c in cac_cau], axis=0)

        if video_id is not None:
            sim = np.where((self.master.video_id == video_id).values, sim, -9.0)
        if chi_co_anh:
            sim = np.where(self.master.kf_path.notna().values, sim, -9.0)

        lay = min(len(sim), k * (moi_video or 1) + 200)
        top = np.argpartition(-sim, lay - 1)[:lay]
        top = top[np.argsort(-sim[top])]
        top = top[sim[top] > -1]

        ra = [Candidate(row_id=int(i), video_id=r.video_id,
                        frame_idx=int(r.frame_idx), score=float(sim[i]),
                        source="clip",
                        meta={"pts_time": float(r.pts_time), "fps": float(r.fps),
                              "kf_n": int(r.kf_n), "title": r.title})
              for i, r in zip(top, self.master.iloc[top].itertuples())]

        if moi_video:
            dem, loc = {}, []
            for c in ra:
                if dem.get(c.video_id, 0) < moi_video:
                    dem[c.video_id] = dem.get(c.video_id, 0) + 1
                    loc.append(c)
            ra = loc
        return ra[:k]


# Truy vấn dùng làm mốc cho bài test cố định ở tests/test_dense.py.
CAU_MOC = "a person riding a motorbike on the street"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=CAU_MOC)
    ap.add_argument("--index", default=Path("./index"), type=Path)
    ap.add_argument("--matrix", default="clip.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ghi-moc", action="store_true",
                    help="ghi mốc hiện tại ra tests/moc_dense.json")
    a = ap.parse_args()

    kenh = KenhAnh(a.index, matrix=a.matrix)
    print(f"{a.matrix}: {kenh.mat.shape}  |  {kenh.model_tag} / {kenh.pretrained}\n")

    kq = kenh.tim(a.query, k=a.k)
    print(f'"{a.query}"\n')
    print(f"{'cos':>6}  {'video_id':<10} {'frame_idx':>9}  {'row_id':>7}  tiêu đề")
    print("-" * 92)
    for c in kq:
        print(f"{c.score:6.3f}  {c.video_id:<10} {c.frame_idx:>9}  {c.row_id:>7}  "
              f"{c.meta['title'][:44]}")

    if a.ghi_moc:
        moc = Path("tests/moc_dense.json")
        moc.parent.mkdir(exist_ok=True)
        dau = kenh.tim(CAU_MOC, k=1)[0]
        moc.write_text(json.dumps({
            "cau": CAU_MOC, "model": kenh.model_tag, "pretrained": kenh.pretrained,
            "chieu": kenh.chieu, "row_id": dau.row_id,
            "video_id": dau.video_id, "cos": round(dau.score, 4),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi mốc -> {moc}")


if __name__ == "__main__":
    main()
