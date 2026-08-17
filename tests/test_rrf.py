"""
test_rrf.py — Chốt cho hợp nhất.

Hợp nhất hỏng không crash: nó trả về một danh sách trông hoàn toàn hợp lý, chỉ
xếp sai. A14 đã cho thấy nó còn có thể LÀM TỆ ĐI mà mọi thứ vẫn chạy êm.
"""

import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from rrf import hop_nhat, hop_nhat_hai_tang, xep_video   # noqa: E402
from schema import Candidate                             # noqa: E402


def c(row_id, video, source="x"):
    return Candidate(row_id=row_id, video_id=video, frame_idx=row_id * 25,
                     score=0.0, source=source)


# ---- RRF thô ---------------------------------------------------------------

def test_dong_thuan_giua_hai_kenh_duoc_thuong():
    """Toàn bộ lý do RRF tồn tại: ứng viên được NHIỀU kênh đề cử phải lên trên.

    row 2 đứng hạng 2 ở CẢ HAI kênh: 2/(60+2) = 0,03226.
    row 1 và row 5 mỗi cái hạng 1 nhưng chỉ ở MỘT kênh: 1/(60+1) = 0,01639.
    Đồng thuận thắng thứ hạng cao hơn — đó là toàn bộ ý tưởng.

    (Ví dụ đầu tôi viết là `[1,2,3]` với `[3,2,1]`: hỏng, vì cả ba dòng đều có
    mặt ở cả hai kênh nên chẳng minh hoạ được đồng thuận. Giữ ghi chú này để
    người sau không dựng lại đúng cái ví dụ vô nghĩa đó.)
    """
    a = [c(1, "V1"), c(2, "V2"), c(3, "V3"), c(4, "V4")]
    b = [c(5, "V5"), c(2, "V2"), c(6, "V6"), c(7, "V7")]
    ra = hop_nhat([a, b])
    assert ra[0].row_id == 2, [x.row_id for x in ra]
    assert ra[0].score > ra[1].score


def test_khong_giao_nhau_thi_chi_dan_xen():
    """A14: hai kênh không chung `row_id` nào thì RRF KHÔNG cộng hưởng được gì,
    nó chỉ đan xen — và đan xen thì đẩy lùi ứng viên tốt của kênh mạnh."""
    a = [c(1, "V1"), c(2, "V2")]
    b = [c(10, "V9"), c(11, "V9")]
    ra = hop_nhat([a, b])
    assert [x.row_id for x in ra] == [1, 10, 2, 11]


def test_trong_so_sai_so_luong_thi_dung_ngay():
    a = [c(1, "V1")]
    try:
        hop_nhat([a, a], trong_so=[1.0])
    except ValueError:
        return
    raise AssertionError("nhận trọng số lệch số lượng mà không báo")


# ---- hai tầng --------------------------------------------------------------

def test_xep_video_theo_khung_tot_nhat():
    ds = [c(1, "V2"), c(2, "V1"), c(3, "V2"), c(4, "V3")]
    assert xep_video(ds) == ["V2", "V1", "V3"]


def test_hai_tang_gom_khung_cua_video_manh_len_truoc():
    """Tầng 1 chọn video, tầng 2 xếp khung trong video đó.

    Cả hai kênh đều để V7 đầu, nên mọi khung của V7 phải lên trước khung của
    video khác — kể cả khung V9 đứng hạng 2 ở một kênh.
    """
    a = [c(1, "V7"), c(9, "V9"), c(2, "V7")]
    b = [c(2, "V7"), c(1, "V7"), c(8, "V9")]
    ra = hop_nhat_hai_tang([a, b], moi_video=2)
    assert {x.video_id for x in ra[:2]} == {"V7"}


def test_hai_tang_ton_trong_moi_video():
    a = [c(i, "V1") for i in range(10)]
    assert len(hop_nhat_hai_tang([a], moi_video=3)) == 3
    assert len(hop_nhat_hai_tang([a], moi_video=99)) == 10


def test_hai_tang_khong_vuot_gioi_han():
    a = [c(i, f"V{i}") for i in range(50)]
    assert len(hop_nhat_hai_tang([a], moi_video=3, gioi_han=7)) == 7


def test_hai_tang_dung_hang_NOI_BO_video():
    """Phải dùng hạng trong nội bộ video, không dùng hạng toàn cục.

    Dùng hạng toàn cục thì video đứng cuối có mọi khung mang hạng rất lớn, và
    thứ tự nội bộ của nó bị nén phẳng thành như nhau. Ở đây V2 đứng sau V1,
    nhưng trong nội bộ V2 thì row 30 vẫn phải trên row 31.
    """
    a = [c(1, "V1"), c(2, "V1"), c(30, "V2"), c(31, "V2")]
    ra = hop_nhat_hai_tang([a], moi_video=2)
    v2 = [x.row_id for x in ra if x.video_id == "V2"]
    assert v2 == [30, 31]


def test_hai_tang_giu_frame_idx_goc():
    """`frame_idx` là giá trị nộp cho BTC — hợp nhất không được tự tính lại."""
    a = [c(5, "V1")]
    assert hop_nhat_hai_tang([a])[0].frame_idx == 125
