"""
test_objects.py — Chốt cho bảng ánh xạ nhãn Việt → Anh.

Bảng này là thứ duy nhất còn thiếu để kênh 4 dùng được từ truy vấn tiếng Việt.
Hỏng nó thì kênh 4 im lặng trả về 0 điểm cho mọi keyframe — không lỗi, không
cảnh báo, chỉ là một kênh chết mà không ai nhận ra.
"""

import sys
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

pytestmark = pytest.mark.skipif(
    not (GOC / "dev" / "label_vi_en.csv").exists(),
    reason="chưa có dev/label_vi_en.csv")


@pytest.fixture(scope="module")
def bang():
    from objects import nap_bang_nhan
    return nap_bang_nhan()


def test_bang_khong_trung_khong_thieu_cot(bang):
    assert not bang.nhan_en.duplicated().any(), "có nhãn lặp lại"
    assert (bang.nhan_vi.str.strip() != "").all(), "có dòng thiếu bản dịch"
    assert (bang.tu.map(len) > 0).all(), "có dòng thiếu từ đồng nghĩa"


def test_moi_nhan_deu_ton_tai_that(bang):
    """Nhãn không có trong `objects.parquet` là nhãn chết — gõ nhầm tên."""
    import pandas as pd
    p = GOC / "index" / "objects.parquet"
    if not p.exists():
        pytest.skip("chưa có index/objects.parquet")
    co = set(pd.read_parquet(p, columns=["label"]).label.unique())
    thua = sorted(set(bang.nhan_en) - co)
    assert not thua, f"nhãn không tồn tại trong kho: {thua}"


def test_cot_cha_tro_dung(bang):
    ten = set(bang.nhan_en)
    xau = sorted({c for c in bang.cha if c and c not in ten})
    assert not xau, f"cột `cha` trỏ tới nhãn không có trong bảng: {xau}"


@pytest.mark.parametrize("cau, phai_co", [
    ("người phụ nữ đang thái cà chua bên chảo", {"Woman", "Tomato", "Frying pan"}),
    ("hai chiếc xe máy màu xanh", {"Motorcycle"}),
    ("chiếc ghe nhỏ trôi trên sông", {"Canoe", "Boat"}),      # đồng nghĩa vùng miền
    ("cán bộ công an ngồi ghi biên bản", {"Person"}),          # từ chỉ nghề nghiệp
    ("hai con mèo rừng nằm trên bãi cỏ", {"Cat", "Animal"}),
    ("bát phở nóng cùng đôi đũa", {"Bowl", "Chopsticks"}),
])
def test_khop_truy_van_that(bang, cau, phai_co):
    from objects import nhan_tu_truy_van
    ra = set(nhan_tu_truy_van(cau, bang))
    assert phai_co <= ra, f"thiếu {phai_co - ra}"


def test_keo_nhan_cha(bang):
    """Thứ bậc OpenImages KHÔNG tự gộp — `Car` không kéo theo `Vehicle`."""
    from objects import nhan_tu_truy_van
    co_cha = set(nhan_tu_truy_van("chiếc ô tô màu đỏ", bang, keo_cha=True))
    khong = set(nhan_tu_truy_van("chiếc ô tô màu đỏ", bang, keo_cha=False))
    assert "Car" in khong
    assert {"Land vehicle", "Vehicle"} <= co_cha
    assert {"Land vehicle", "Vehicle"} & khong == set()


def test_khop_theo_cum_tu_khong_phai_chuoi_con(bang):
    """"cá" nằm trong "cá nhân" — khớp chuỗi con là sai.

    Khớp n-gram thì "cá nhân" tách thành ["cá", "nhân"] nên 1-gram "cá" VẪN
    khớp. Đây là nhập nhằng của chính tiếng Việt, chấp nhận được vì
    `object_score` cho điểm mềm. Nhưng chuỗi con dính ngay cả khi KHÔNG có
    ranh giới từ — ca đó phải chặn được.
    """
    from objects import nhan_tu_truy_van
    # "bàn" là nhãn Table; "bànphím" viết liền không được coi là "bàn"
    assert "Table" not in nhan_tu_truy_van("bànphímmáytính", bang)
    assert "Table" in nhan_tu_truy_van("cái bàn gỗ", bang)
