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

ĐÃ ĐO — CHƯA ĐƯỢC VÀO ĐƯỜNG ỐNG MẶC ĐỊNH
=========================================

`scripts/13_do_dedup.py`, 97 câu tập dev + 15 truy vấn tiếng Anh đối chứng.

    truy vấn TIẾNG ANH (CLIP đọc được)   bỏ  0,5/100   lệch  2,6/100 vị trí
    truy vấn TIẾNG VIỆT (CLIP mù)        bỏ 58,4/100   lệch 77,5/100 vị trí

Con số 58,4 KHÔNG phải lợi ích của dedup. CLIP mù tiếng Việt (A10) nên vector
truy vấn gần như ngẫu nhiên, top-100 đổ dồn vào một cảnh tĩnh, và dedup dọn
đống đó. Nó đo cái hỏng của kênh 1. Số dùng được là 0,5/100.

Vì sao nhỏ đến vậy dù kho có 11,83% trùng lặp: bản sao chỉ tính TRONG CÙNG
video, mà top-100 của một truy vấn đọc được trải ra 63 video khác nhau — mỗi
video góp một hai frame, hiếm khi hai thành viên cùng cụm gặp nhau trong cùng
top-100. Ép bể chỉ còn L25 (49,82% trùng lặp, tệ nhất kho) cũng chỉ bỏ 2,5/100.

Và **HẠNG 1 KHÔNG ĐỔI Ở BẤT KỲ PHÉP ĐO NÀO** — 0/97 và 0/15 câu. R@1 chiếm 1/5
tổng điểm, dedup không chạm tới nó.

Chỗ duy nhất còn đất: KHI BẬT `moi_video`. Ràng buộc đó giữ tối đa k frame mỗi
video, và nếu k frame ấy là k bản sao thì cả ngân sách của video đó phí. Đo
trên bể L25 với moi_video=3: lệch 30,5/100 vị trí (toàn kho: 2,3). Đúng như
docstring dưới đây dự đoán — `gioi_han_moi_video()` không thay được dedup, hai
cái BỔ SUNG cho nhau.

CHƯA KẾT LUẬN ĐƯỢC VỀ ĐIỂM, và phải nói rõ vì sao: kênh duy nhất đọc được
tiếng Việt là SigLIP2, mà ma trận thử của nó mới encode 11 video L21+L22 —
đúng hai nhóm TRÙNG LẶP ÍT NHẤT KHO (0,45% và 0,27%). Đo dedup ở đó là đo chỗ
nó không có gì để làm.

    => Giữ module. KHÔNG bật mặc định. Đo lại khi có `clip_siglip2.npy` toàn
       kho, trên câu L25, với `moi_video` bật:

           python scripts/13_do_dedup.py --matrix clip_siglip2.npy --moi-video 3
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
    # ⚠️ 11,83% của toàn kho là con số ĐÁNH LỪA: 18.654 trong 20.975 bản sao
    # (89%) nằm ở riêng L25. Chín nhóm còn lại đều dưới 2,2%. Đừng suy từ
    # 11,83% ra "kho nào cũng vậy".
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
