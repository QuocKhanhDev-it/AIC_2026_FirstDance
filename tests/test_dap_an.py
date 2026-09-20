"""
test_dap_an.py — Chốt cho việc điền `answer` từng dòng (A67).

Ba thứ đáng chốt, và cả ba đều hỏng ÂM THẦM:
  * mọi dòng nhận CÙNG một chuỗi -> vứt đi cơ chế chấm từng dòng của BTC
  * lấy ứng viên ĐẦU TIÊN -> trúng đồng hồ "06:30:11" của ticker bản tin
  * dòng không đào được gì mà để TRỐNG -> chắc chắn 0, trong khi đáp án sai
    không bị phạt gì thêm
"""

import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from dap_an import dao, gan_cho_moi_dong  # noqa: E402
from schema import Candidate  # noqa: E402


def _uv(*rid):
    return [Candidate(row_id=r, video_id="L21_V001", frame_idx=r * 10,
                      score=1.0, source="t", meta={}) for r in rid]


def test_chon_so_gan_tu_khoa_khong_lay_so_dau():
    """OCR bản tin mở đầu bằng đồng hồ. 'Số đầu tiên' gần như luôn là giờ
    phát sóng — đúng cái làm bộ đào ngây thơ trượt 13/13 câu."""
    van = "06:30:11 HTV9 TIN CHÍNH hôm nay thu hoạch được 46 tấn lúa"
    assert dao(van, "Nông dân thu hoạch được bao nhiêu tấn lúa?") == "46"


def test_hoi_ten_thi_lay_ten_rieng():
    van = "Phóng viên Giang Ly tường thuật từ hiện trường"
    assert dao(van, "Tên phóng viên xuất hiện trên màn hình là gì?") == "Giang Ly"


def test_khong_co_gi_thi_tra_chuoi_rong():
    assert dao("", "bao nhiêu?") == ""
    assert dao("HTV9", "Tên người dẫn là gì?") == ""


def test_moi_dong_mot_dap_an_khac_nhau():
    """Đây là cả điểm của module: BTC chấm answer THEO TỪNG DÒNG."""
    van = {1: "thu được 46 tấn", 2: "thu được 88 tấn", 3: ""}
    uv = _uv(1, 2, 3)
    gan_cho_moi_dong(uv, "thu được bao nhiêu tấn?", van, mac_dinh="không rõ")
    assert [c.meta["answer"] for c in uv] == ["46", "88", "không rõ"]


def test_dong_khong_dao_duoc_van_co_chuoi():
    """Để trống là chắc chắn 0 điểm, mà đáp án sai không bị phạt thêm."""
    uv = _uv(9)
    gan_cho_moi_dong(uv, "bao nhiêu?", {}, mac_dinh="không rõ")
    assert uv[0].meta["answer"] == "không rõ"


def test_dem_dung_so_dong_dao_duoc():
    van = {1: "46 tấn", 2: "", 3: "88 tấn"}
    n = gan_cho_moi_dong(_uv(1, 2, 3), "bao nhiêu tấn?", van)
    assert n == 2


def test_nop_bai_uu_tien_answer_tung_dong():
    """`tu_ung_vien` phải lấy `meta['answer']` chứ không phải chuỗi chung."""
    from nop_bai import tu_ung_vien
    uv = _uv(1, 2)
    uv[0].meta["answer"] = "46"
    uv[1].meta["answer"] = "88"
    ra = tu_ung_vien(uv, "qa", dap_an="CHUNG")
    assert [x.answer for x in ra] == ["46", "88"]


def test_uu_tien_ban_co_dau_khi_ca_hai_cung_co_mat():
    """OCR khong dau + ASR co dau trong CUNG mot chuoi -> phai lay ban co dau.

    A68: ocr_text 31% co dau, asr_text 100%; run.py ghep `OCR + " " + ASR` nen
    cung mot thuc the xuat hien hai lan. BTC khop CHUOI nen `Ta Pua` la 0 diem.

    Fixture dat ban KHONG DAU ngay canh tu khoa cau hoi, de phep chon theo
    khoang cach chac chan lay no truoc — co the uu tien co dau moi phai lam
    viec, chu khong an may vi tri.
    """
    from dap_an import dao
    van = "Ta Pua vua hoan thanh con duong. Ban tin hom nay tai Tà Pứa."
    assert dao(van, "Địa danh nào vừa hoàn thành con đường") == "Tà Pứa"


def test_khong_tu_bia_dau_khi_ban_co_dau_khong_ton_tai():
    """Chi CHON dung ban, KHONG doan dau — khong co ban co dau thi giu nguyen."""
    from dap_an import dao
    van = "Ta Pua vua hoan thanh con duong. Ban tin hom nay."
    assert dao(van, "Địa danh nào vừa hoàn thành con đường") == "Ta Pua"


def test_co_dau_nhan_dien_dung():
    from dap_an import co_dau
    assert co_dau("Tà Pứa")
    assert co_dau("Đường")           # dau mu cung tinh
    assert not co_dau("Ta Pua")
    assert not co_dau("06:30")
