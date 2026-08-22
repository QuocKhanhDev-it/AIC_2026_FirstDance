"""
cua_so.py — Chấm điểm theo CỬA SỔ keyframe thay vì theo từng keyframe rời.

VÌ SAO — A38, rút từ ghi chép soát tay của nhóm
================================================

Đề thi được viết bằng cách **xem video**; hệ thống lại tìm trên **keyframe**.
Người viết đề mô tả một quãng thời gian, ta đi khớp từng ảnh tĩnh rời rạc. Một
truy vấn 63 từ / 2,4 mệnh đề gần như **không bao giờ** có đủ ngữ nghĩa trong một
keyframe duy nhất.

Ca `query-p1-4-kis` là ví dụ sạch nhất:

    kf183  "đàn sư tử ... bảng London Zoo"        <- chỉ mệnh đề 1
    kf186  "nhân viên áo xanh cân thú"            <- chỉ mệnh đề 2, MỘT người
    kf187  "nhân viên áo xanh cân thú"            <- cũng MỘT người

Không khung nào phủ cả hai mệnh đề. Người soát cộng chúng lại mới ra *"hai nhân
viên"* như đề tả.

CÁCH TÍNH
=========

    điểm(khung f) = Σ_j  max_{g ∈ cửa sổ quanh f}  sim(g, mệnh_đề_j)

Mỗi mệnh đề đi tìm khung tốt nhất **trong vùng lân cận của f**, rồi cộng lại.
Khung nào có cả cụm quanh nó phủ được nhiều mệnh đề thì thắng.

Khác chỗ nào với bản hiện hành: `dense.KenhAnh.tim` lấy `max` **qua các mệnh đề
trên cùng một khung**, tức thưởng cho khung khớp MỘT mệnh đề thật mạnh. Ở đây
**cộng** qua mệnh đề và lấy `max` **qua các khung lân cận** — thưởng cho vùng
phủ được nhiều mệnh đề.

⚠️ **KHÔNG mâu thuẫn với A18.** A18 bác việc *chèn khung lân cận vào 100 dòng
nộp* — ở đó lân cận **tiêu mất chỗ**. Ở đây lân cận chỉ dùng để **chấm điểm**;
dòng nộp vẫn là khung f. Cùng một dữ liệu, đổi vai trò thì đổi giá trị — như
kênh OCR ở A27/A28.

⚠️ **Cửa sổ không được vượt sang video khác.** `master.parquet` sắp theo
`(video_id, frame_idx)` nên keyframe liền kề chỉ là `row_id ± 1` — nhưng ở biên
video thì hàng xóm là video khác. Phải chặn.
"""

import numpy as np
import pandas as pd


def bien_video(master: pd.DataFrame) -> np.ndarray:
    """Mảng cùng độ dài bảng cái: chỉ số nhóm video của từng dòng.

    Dùng để chặn cửa sổ tràn sang video khác mà không phải tra `video_id` cho
    từng dòng (177.321 lần tra chuỗi là chậm thật, đã đo).
    """
    ma, _ = pd.factorize(master.video_id.values)
    return ma


def diem_cua_so(sim: np.ndarray, nhom: np.ndarray, ban_kinh: int = 3) -> np.ndarray:
    """`sim` là (số_mệnh_đề, số_dòng) -> điểm cửa sổ cho từng dòng.

    Với mỗi mệnh đề, lấy max trượt trong cửa sổ `±ban_kinh` dòng **cùng video**,
    rồi cộng qua các mệnh đề.

    Cài bằng phép dịch mảng thay vì vòng lặp Python: 177.321 dòng × vài mệnh đề
    × vài chục lần dịch vẫn dưới một giây, còn vòng lặp thì hàng phút.
    """
    sim = np.atleast_2d(np.asarray(sim, dtype=np.float32))
    n = sim.shape[1]
    if n != len(nhom):
        raise ValueError(f"sim có {n} dòng nhưng nhóm có {len(nhom)}")

    tong = np.zeros(n, dtype=np.float32)
    for hang in sim:
        tot = hang.copy()
        for d in range(1, ban_kinh + 1):
            # dịch phải: dòng i nhìn sang i-d, chỉ khi cùng video
            cung = nhom[d:] == nhom[:-d]
            tot[d:] = np.where(cung, np.maximum(tot[d:], hang[:-d]), tot[d:])
            # dịch trái: dòng i nhìn sang i+d
            tot[:-d] = np.where(cung, np.maximum(tot[:-d], hang[d:]), tot[:-d])
        tong += tot
    return tong


def diem_khung_roi(sim: np.ndarray) -> np.ndarray:
    """Cách hiện hành, để làm mốc nền: max qua các mệnh đề trên cùng một khung."""
    return np.max(np.atleast_2d(np.asarray(sim, dtype=np.float32)), axis=0)
