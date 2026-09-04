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


def test_mac_dinh_la_300():
    """A102: 300 là mặc định, và đây là lần đổi CÓ CĂN CỨ VƯỢT NGƯỠNG.

    A94 đo 300 so với mốc `kênh 1+3, bể 100` được +0,0739 -> 🟡, nên lúc đó
    mặc định để None. A102 phát hiện mốc đó **không phải thứ bài nộp chạy**:
    bài nộp chạy TRAKE bằng kênh 1 một mình. So với mốc THẬT thì 300 được
    **+0,1244/+0,1000 ✅ ỔN ĐỊNH**.

    ⚠️ Vẫn còn một cảnh báo phải nhớ: **15/18 câu TRAKE đo được là TỰ SOẠN**,
    chỉ 3 câu là đề thật. Đổi mặc định này lần nữa thì phải có nhãn TRAKE đề
    thật, không phải thêm một phép dò tham số.
    """
    m = re.search(r'"--be-trake",\s*type=int,\s*default=(\w+)', NGUON)
    assert m, "không tìm thấy khai báo --be-trake"
    assert m.group(1) == "300", (
        "mặc định phải là 300 (A102, ✅ +0,1244/+0,1000 so với mốc nền thật). "
        "Đổi thì phải sửa cả docstring test này kèm phép đo mới.")


def test_trake_duoc_hop_nhat_kenh_3():
    """A102: bài nộp TỪNG chạy TRAKE bằng kênh 1 một mình — chốt để không tái phạm.

    `quet_van_ban` bỏ qua TRAKE và `phu[ten]` chỉ được dùng ở nhánh
    không-TRAKE, nên suốt một thời gian dài mọi kết luận TRAKE của repo đo
    trên một cấu hình bài nộp không chạy. Đo được khoảng cách: **0,2994 ->
    0,4317 ở ±2s**.
    """
    assert 'ra[ten] = [k3.tim(sk, k=k) for sk in tach_su_kien(nd)]' in NGUON, \
        "quet_van_ban không còn sinh ứng viên kênh 3 cho TRAKE"
    assert "if a.hop_nhat and p3 and len(p3) == len(ds):" in NGUON, \
        "nhánh TRAKE không còn hợp nhất kênh 3 vào từng sự kiện"


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
