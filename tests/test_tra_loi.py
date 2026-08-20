"""
test_tra_loi.py — Chốt cho bộ sinh `answer` của câu Q&A.

Phần KHÔNG cần Ollama: dọn đáp án. Đây là chỗ quyết định điểm — BTC chấm
`answer` bằng chuỗi, nên chữ thừa là mất trắng câu chứ không phải trừ điểm.
"""

import sys
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from tra_loi import TOI_DA, don_dap_an        # noqa: E402


@pytest.mark.parametrize("tho, mong", [
    ("Trả lời: 5", "5"),
    ("Đáp án: màu đỏ", "màu đỏ"),
    ("Answer: ba người", "ba người"),
    ("Trong ảnh: hai con mèo", "hai con mèo"),
    ("Dựa vào hình: nón lá", "nón lá"),
    ("Có 3 người", "3 người"),
    ("  màu xanh lá.  ", "màu xanh lá"),
])
def test_bo_chu_dua_day(tho, mong):
    """Model hay thêm mở đầu dù đã bảo đừng. `"Trả lời: 5"` khác `"5"` khi BTC
    so chuỗi — mất trắng câu."""
    assert don_dap_an(tho) == mong


def test_bo_nhieu_lop_mo_dau():
    assert don_dap_an("Trả lời: Đáp án: 7") == "7"


def test_cau_hoi_dem_giu_lai_con_so():
    """*"5 cái bát màu trắng trên bàn gỗ"* -> `"5"`. Con số là phần BTC chấm."""
    assert don_dap_an("5 cái bát màu trắng đặt trên bàn gỗ") == "5"


def test_khong_cat_khi_con_so_di_kem_it_chu():
    """Chỉ rút gọn khi phần sau rõ ràng là lặp lại câu hỏi. `"3 người"` giữ
    nguyên — cắt thành `"3"` là đổi nghĩa."""
    assert don_dap_an("3 người") == "3 người"


def test_ep_duoi_100_ky_tu_va_cat_o_ranh_gioi_tu():
    """BTC chặn cứng 100 ký tự — quá là hỏng cả dòng, không phải mất một phần."""
    t = don_dap_an("một " * 60)
    assert len(t) <= TOI_DA
    assert not t.endswith("mộ"), "cắt giữa từ"


def test_chi_lay_dong_dau():
    """Model nhả nhiều dòng suy luận thì chỉ dòng đầu là câu trả lời."""
    assert don_dap_an("màu vàng\nVì tôi thấy chiếc áo...") == "màu vàng"


def test_rong_van_ra_rong_khong_no_loi():
    assert don_dap_an("") == ""
    assert don_dap_an(None) == ""
    assert don_dap_an("   ") == ""


def test_khong_ro_giu_nguyen():
    """VLM không nhìn ra thì phải nói không rõ — giữ nguyên, đừng bịa thêm."""
    assert don_dap_an("không rõ") == "không rõ"


def test_dap_an_di_qua_duoc_bo_soat_nop_bai():
    """Chốt nối: thứ `don_dap_an` sinh ra phải được `nop_bai.soat` chấp nhận."""
    from nop_bai import soat
    from schema import AnswerQA
    tra = don_dap_an("Trả lời: " + "rất dài " * 40)
    assert not soat("query-1-qa", [AnswerQA("L01_V001", 100, tra)])
