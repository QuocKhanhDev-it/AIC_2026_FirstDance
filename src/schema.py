"""
schema.py — Kiểu dữ liệu dùng chung giữa các kênh.

Mọi kênh truy hồi trả về `list[Candidate]`. Nhờ vậy hàm RRF ở `rrf.py` không
cần biết kênh nào sinh ra danh sách nào — thêm kênh thứ 6 về sau không phải
sửa dòng nào trong RRF.

⚠️ `frame_idx` là giá trị NỘP CHO BTC. Đừng bao giờ tự tính lại từ `pts_time`:
làm tròn lệch 1 frame (đo thật: 1061,36 × 25fps = 26534, giá trị đúng 26533).
Luôn lấy thẳng từ cột `frame_idx` của bảng cái.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """Một khoảnh khắc ứng viên do một kênh đề xuất.

    `score` chỉ có ý nghĩa TRONG NỘI BỘ một kênh. Không so điểm giữa các kênh
    với nhau — cosine CLIP nằm khoảng 0,25–0,40 còn BM25 không chặn trên. Đó
    chính là lý do hợp nhất bằng RRF (theo THỨ HẠNG) chứ không phải cộng có
    trọng số.
    """

    row_id: int
    video_id: str
    frame_idx: int
    score: float
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerKIS:
    video_id: str
    frame_idx: int


@dataclass
class AnswerQA:
    video_id: str
    frame_idx: int
    answer: str          # sai định dạng -> 0 điểm dù frame đúng (PHẦN C mục 4)


@dataclass
class AnswerTRAKE:
    video_id: str
    frame_idxs: list[int]
    # PHẦN C mục 3: điểm tính TỪNG PHẦN theo số sự kiện khớp. Bỏ trống chắc
    # chắn 0, đoán sai cũng 0 -> LUÔN điền đủ N vị trí, đừng để thiếu.
