"""
test_thoi_gian.py — Chốt cho xếp hạng lại theo chuỗi TRAKE (Bước 2b).
"""

import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from schema import Candidate                              # noqa: E402
from thoi_gian import video_du_chuoi, xep_video_theo_chuoi  # noqa: E402


def c(row_id, video, score):
    return Candidate(row_id=row_id, video_id=video, frame_idx=row_id * 25,
                     score=score, source="x")


def test_video_du_chuoi_doi_giao_cung():
    """Bài học cũ, giữ nguyên: thiếu ở MỘT sự kiện là loại cả video."""
    su_kien_1 = [c(1, "V1", 0.9), c(2, "V2", 0.8)]
    su_kien_2 = [c(3, "V2", 0.7)]           # V1 vắng mặt ở sự kiện 2
    assert video_du_chuoi([su_kien_1, su_kien_2]) == ["V2"]


def test_xep_video_theo_chuoi_khong_doi_giao_cung():
    """V1 thiếu ở sự kiện 2 vẫn phải có mặt trong kết quả — không bị loại."""
    su_kien_1 = [c(1, "V1", 0.9), c(2, "V2", 0.8)]
    su_kien_2 = [c(3, "V2", 0.7)]
    ra = xep_video_theo_chuoi([su_kien_1, su_kien_2])
    assert set(ra) == {"V1", "V2"}


def test_xep_video_theo_chuoi_thuong_video_du_chuoi_lan_can():
    """V2 khớp cả 3 sự kiện (được thưởng điểm trước+sau ở sự kiện giữa),
    V1 chỉ khớp 1 sự kiện — V2 phải xếp trên, dù điểm gốc mỗi sự kiện bằng
    nhau."""
    su_kien_1 = [c(1, "V1", 0.5), c(2, "V2", 0.5)]
    su_kien_2 = [c(3, "V2", 0.5)]
    su_kien_3 = [c(4, "V2", 0.5)]
    ra = xep_video_theo_chuoi([su_kien_1, su_kien_2, su_kien_3])
    assert ra[0] == "V2"


def test_xep_video_theo_chuoi_mot_su_kien():
    """Chuỗi 1 sự kiện (N=1): không có trước/sau, vẫn phải chạy được."""
    su_kien_1 = [c(1, "V1", 0.9), c(2, "V2", 0.5)]
    assert xep_video_theo_chuoi([su_kien_1]) == ["V1", "V2"]


def test_xep_video_theo_chuoi_rong():
    assert xep_video_theo_chuoi([]) == []
