"""
lan_can.py — Đi bộ theo thời gian quanh một kết quả ("nearby frame").

Theo A8.7 đây là kỹ thuật có tỷ lệ lợi/công cao nhất lấy từ bài báo AIC'25.
Nó xuất hiện ở 2/5 ví dụ thực chiến của họ và là bước xoay chuyển của cả câu
Q&A lẫn câu TRAKE khó nhất:

  * Câu Q&A hỏi "chiếc bánh được cắt làm mấy phần". Gợi ý cho biết NGUYÊN LIỆU,
    còn đáp án nằm ở BƯỚC CẮT vài giây sau. Truy hồi ngữ nghĩa đưa tới lân cận;
    đi bộ đưa tới đích.
  * Câu TRAKE: lọc OCR bảng tỷ số ra một khung NGAY SAU bàn thắng, rồi đi
    ngược lại để tìm đúng khoảnh khắc bóng qua vạch.

Rẻ vì `master.parquet` đã sắp theo `(video_id, frame_idx)` VÀ `row_id` tăng
dần cùng chiều — nên keyframe liền kề trong một video chỉ là `row_id ± 1`.
Không cần tra cứu, không cần sắp lại.

Dùng:
    from lan_can import lan_can, lan_can_nhieu
    quanh = lan_can(master, row_id=12345, so_buoc=10)
"""

import numpy as np
import pandas as pd

try:
    from .schema import Candidate
except ImportError:
    from schema import Candidate


def bien_video(master: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Khoảng `row_id` [đầu, cuối] của từng video. Tính một lần, dùng lại."""
    g = master.groupby("video_id")["row_id"].agg(["min", "max"])
    return {v: (int(a), int(b)) for v, a, b in g.itertuples()}


def lan_can(master: pd.DataFrame, row_id: int, so_buoc: int = 10,
            bien: dict | None = None) -> list[Candidate]:
    """Các keyframe liền kề theo thời gian, KHÔNG vượt sang video khác.

    Trả về cả hai phía, sắp theo thời gian tăng dần, có kèm chính `row_id` gốc.
    `meta['buoc']` là khoảng cách tương đối: âm = trước, 0 = chính nó, dương =
    sau. `meta['cach_giay']` là chênh lệch thời gian thật.
    """
    r = master.iloc[row_id]
    bien = bien or bien_video(master)
    dau, cuoi = bien[r.video_id]

    lo = max(dau, row_id - so_buoc)
    hi = min(cuoi, row_id + so_buoc)
    lat = master.iloc[lo:hi + 1]

    return [Candidate(row_id=int(x.row_id), video_id=x.video_id,
                      frame_idx=int(x.frame_idx), score=0.0, source="lan_can",
                      meta={"buoc": int(x.row_id) - row_id,
                            "cach_giay": round(float(x.pts_time) - float(r.pts_time), 2),
                            "pts_time": float(x.pts_time), "fps": float(x.fps),
                            "kf_n": int(x.kf_n)})
            for x in lat.itertuples()]


def lan_can_nhieu(master: pd.DataFrame, ung_vien: list[Candidate],
                  so_buoc: int = 5, toi_da: int = 500) -> list[Candidate]:
    """Mở rộng một danh sách ứng viên ra vùng lân cận của từng cái.

    Dùng khi nghi đáp án nằm gần kết quả truy hồi chứ không phải đúng nó —
    tình huống điển hình của Q&A. Bỏ trùng, giữ nguyên thứ tự xuất hiện.

    ⚠️ Nở danh sách ra `so_buoc × 2 + 1` lần. Đừng nhét thẳng vào RRF: RRF đọc
    THỨ HẠNG, mà các khung lân cận vào đây đều mang score 0 và thứ hạng giả.
    Hãy chấm lại chúng bằng một kênh thật (CLIP, VLM) trước khi hợp nhất.
    """
    bien = bien_video(master)
    thay, ra = set(), []
    for c in ung_vien:
        for x in lan_can(master, c.row_id, so_buoc, bien):
            if x.row_id not in thay:
                thay.add(x.row_id)
                ra.append(x)
                if len(ra) >= toi_da:
                    return ra
    return ra


def cung_frame_idx(master: pd.DataFrame, row_id: int) -> list[int]:
    """Các `row_id` khác CÙNG video CÙNG `frame_idx` với dòng đã cho.

    A5.7 đo được 614 keyframe (0,35%, ở 192 video) có `frame_idx` trùng hệt
    dòng liền trước. Không phép đo nào phân biệt được chúng — nhưng cũng không
    cần: theo A8.1 điểm chấm theo KHOẢNG nên nộp giá trị đó là trúng cả hai.
    Hàm này chỉ để soi, không dùng khi nộp bài.
    """
    r = master.iloc[row_id]
    cung = master[(master.video_id == r.video_id)
                  & (master.frame_idx == r.frame_idx)]
    return [int(x) for x in cung.row_id if int(x) != row_id]
