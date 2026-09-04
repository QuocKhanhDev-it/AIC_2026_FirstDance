"""
test_be_trake.py — Chốt cho `--be-trake` (A94).

Cờ này đổi BỂ ỨNG VIÊN của TRAKE. Nó mới đạt 🟡 nên mặc định phải là "không
đổi gì"; và vì `be_trake or k` dùng toán tử `or`, giá trị 0 sẽ lặng lẽ rơi về
`k` — một cái bẫy đáng chốt lại.
"""

import re
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))
NGUON = (GOC / "src" / "run.py").read_text("utf-8")


def test_mac_dinh_khong_doi_gi():
    """A94 mới 🟡 (+0,0739 so với ngưỡng 0,0876) nên KHÔNG được bật mặc định."""
    m = re.search(r'"--be-trake",\s*type=int,\s*default=(\w+)', NGUON)
    assert m, "không tìm thấy khai báo --be-trake"
    assert m.group(1) == "None", (
        "mặc định phải là None (= dùng --k, tức giữ nguyên hành vi cũ). "
        "Muốn bật thì phải có phép đo vượt ngưỡng trước — và nhớ rằng 15/18 "
        "câu TRAKE đo được là TỰ SOẠN.")


def test_chi_trake_dung_be_rieng_kis_va_qa_giu_nguyen():
    """Chỉ nhánh TRAKE được dùng bể riêng; KIS/Q&A vẫn `k * 2` như cũ.

    Nếu bể riêng lọt sang nhánh KIS/Q&A thì nó chỉ làm chậm mà không đổi điểm
    (chỉ nộp được 100 dòng), và tệ hơn là làm mọi phép đo KIS cũ không so lại
    được.
    """
    assert "ra[ten] = [hoi(sk, be_trake or k) for sk in tach_su_kien" in NGUON
    assert "ra[ten] = hoi(noi_dung, k * 2)" in NGUON, \
        "nhánh KIS/Q&A phải giữ nguyên `k * 2`"


def test_be_trake_duoc_noi_qua_ca_hai_duong_goi():
    """`quet_anh` có hai chỗ gọi (giữ kênh / không giữ kênh) — cả hai phải nối.

    Quên một đường là cờ im lặng vô hiệu ở đúng đường đó — đúng loại lỗi mà
    commit `8a27e29` đã sửa bốn lần.
    """
    assert NGUON.count("be_trake=a.be_trake") == 2, (
        "phải nối `be_trake` ở CẢ HAI chỗ gọi quet_anh, "
        f"đang thấy {NGUON.count('be_trake=a.be_trake')}")
