"""
dedup.py — Gộp keyframe gần trùng trong CÙNG một video trước khi cắt top-K.

Vì sao cần (A5.6 + PHẦN C mục 6): 11,83% keyframe của kho có một bản sao cùng
video ở cosine ≥ 0,99 — riêng L25 là 49,82%, và có cặp cosine đúng bằng 1,0000.
Điểm thi tính `max R-Score trong top-k`, mà `R@1` chiếm 1/5 tổng điểm. Nếu
top-5 bị năm bản sao gần như y hệt của cùng một khoảnh khắc chiếm chỗ thì ta
phí 4 slot mà không tăng chút cơ hội trúng nào.

Ràng buộc đa dạng ở `rrf.gioi_han_moi_video()` KHÔNG thay được việc này: nó
đếm theo `video_id` nên hai bản sao trong cùng một video vẫn lọt qua.

Thứ tự đúng trong đường ống:

    kênh 1..5  ->  rrf.hop_nhat  ->  dedup.gom_ban_sao  ->  gioi_han_moi_video
                                     ^^^^^^^^^^^^^^^^^
                            đặt ở đây: sau khi đã xếp hạng, trước khi cắt

Đặt sớm hơn (ngay trong từng kênh) sẽ vứt mất ứng viên mà kênh khác còn cần.

Dùng:
    from dedup import gom_ban_sao
    sach = gom_ban_sao(ung_vien, kenh.mat)

⚠️ Việc này chưa được chứng minh là tăng điểm. Theo kỷ luật GIAI ĐOẠN 3, đo
trên tập dev của Khánh: không tăng thì bỏ.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .schema import Candidate
except ImportError:
    from schema import Candidate

# Cùng ngưỡng dùng ở 03_verify_CLIP.py và khi dựng trung_lap.parquet. Giữ khớp
# để các con số trong tài liệu so sánh được với nhau.
NGUONG = 0.99


def gom_ban_sao(ung_vien: list[Candidate], ma_tran, nguong: float = NGUONG,
                giu_moi_cum: int = 1) -> list[Candidate]:
    """Bỏ bớt bản sao gần trùng, giữ ứng viên ĐIỂM CAO NHẤT của mỗi cụm.

    Duyệt theo thứ tự điểm giảm dần; một ứng viên bị loại nếu nó giống một
    ứng viên ĐÃ GIỮ của **cùng video** ở mức ≥ `nguong`.

    Chỉ so trong cùng video: hai keyframe giống nhau ở hai video KHÁC nhau là
    thông tin thật (cùng một tin được hai đài đưa), không phải dư thừa — và
    giữ cả hai còn làm tăng cơ hội trúng video đúng.

    `giu_moi_cum > 1` để giữ vài đại diện mỗi cụm. Chỉ cần nếu BTC đổi luật
    sang chấm `frame_idx` chính xác; theo A8.1 thì luật hiện hành chấm theo
    KHOẢNG nên 1 là đủ.
    """
    if not ung_vien:
        return []

    ung_vien = sorted(ung_vien, key=lambda c: -c.score)
    rid = np.array([c.row_id for c in ung_vien])
    vec = np.asarray(ma_tran[rid], dtype=np.float32)

    giu_theo_video: dict[str, list[int]] = {}     # video_id -> vị trí đã giữ
    ra = []
    for i, c in enumerate(ung_vien):
        da_giu = giu_theo_video.get(c.video_id, [])
        if da_giu:
            cos = vec[da_giu] @ vec[i]
            if int((cos >= nguong).sum()) >= giu_moi_cum:
                continue
        giu_theo_video.setdefault(c.video_id, []).append(i)
        ra.append(c)
    return ra


def thong_ke(index_dir="./index", nguong: float = NGUONG) -> pd.DataFrame:
    """Tỷ lệ keyframe có bản sao cùng video, theo nhóm L.

    Đọc `index/trung_lap.parquet` (cột `max_cos` = cosine tới keyframe giống
    nhất trong cùng video). Dùng để soi lại A5.6, không dùng khi truy hồi.
    """
    d = pd.read_parquet(Path(index_dir) / "trung_lap.parquet")
    g = d.groupby("nhom").agg(keyframe=("row_id", "size"),
                              trung_lap=("max_cos", lambda s: int((s >= nguong).sum())))
    g["ty_le_%"] = (g.trung_lap / g.keyframe * 100).round(2)
    tong = pd.DataFrame({"keyframe": [len(d)],
                         "trung_lap": [int((d.max_cos >= nguong).sum())]},
                        index=["TỔNG"])
    tong["ty_le_%"] = (tong.trung_lap / tong.keyframe * 100).round(2)
    return pd.concat([g.sort_values("ty_le_%", ascending=False), tong])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./index")
    ap.add_argument("--nguong", type=float, default=NGUONG)
    a = ap.parse_args()
    print(f"Keyframe có bản sao cùng video ở cosine >= {a.nguong}\n")
    print(thong_ke(a.index, a.nguong).to_string())


if __name__ == "__main__":
    main()
