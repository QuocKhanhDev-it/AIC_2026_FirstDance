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


@pytest.mark.parametrize("cau, khong_duoc_co", [
    ("cái ấm đun nước", "Cabbage"),        # "cai" <- "cải", không phải "cái"
    ("bàn tay người cầm dao", "Orange"),   # "cam" <- "cầm", không phải quả cam
    ("ô tô màu đỏ", "Boat"),               # "do"  <- "đò",  không phải "đỏ"
    ("bàn tay người cầm dao", "Table"),    # cụm dài "bàn tay" phải nuốt "bàn"
])
def test_co_dau_thi_khong_duoc_nhap_nhem(bang, cau, khong_duoc_co):
    """Bỏ dấu vô điều kiện là GỘP NHẦM những từ khác hẳn nhau.

    Tiếng Việt dùng dấu để phân biệt từ. Ba ca dưới đây đều là lỗi thật đã đo
    được khi bỏ dấu vô điều kiện. Truy vấn CÓ dấu phải khớp có dấu.
    """
    from objects import nhan_tu_truy_van
    assert khong_duoc_co not in nhan_tu_truy_van(cau, bang)


@pytest.mark.parametrize("cau, phai_co", [
    ("ca chua va chao", {"Tomato", "Wok"}),
    ("can bo cong an", {"Person"}),
    ("xe may tren duong", {"Motorcycle"}),
    ("trai thom chin", {"Pineapple"}),
])
def test_khong_dau_van_phuc_vu_duoc(bang, cau, phai_co):
    """Người gõ nhanh trong phiên thi hay bỏ dấu — và đề thi cũng có thể ra
    không dấu (bài báo AIC'25 có truy vấn "giai phong khi hidro")."""
    from objects import nhan_tu_truy_van
    ra = set(nhan_tu_truy_van(cau, bang))
    assert phai_co <= ra, f"thiếu {phai_co - ra}"


@pytest.mark.parametrize("cau, khong_duoc_co", [
    ("mùi thơm của món ăn", "Pineapple"),      # "thơm" = có mùi dễ chịu
    ("bản cam kết", "Orange"),                 # "cam" trong "cam kết"
    ("người dân cam chịu", "Orange"),
    ("chỗ trống trong phòng", "Drum"),         # "trống" = rỗng
    ("chia ly", "Wine glass"),                 # "ly" = rời xa
])
def test_tu_da_nghia_khong_keo_nham(bang, cau, khong_duoc_co):
    """Từ đa nghĩa: chỉ giữ cụm ĐÃ KHỬ NHẬP NHẰNG, bỏ dạng trần.

    Đây là chỗ ngược với trực giác "tách cụm từ ghép cho dễ khớp". Với từ đa
    nghĩa thì **cụm có từ loại mới là thứ khử nhập nhằng**: giữ "trái thơm",
    "quả cam", "dàn trống", "ly nước" — bỏ "thơm", "cam", "trống", "ly".
    """
    from objects import nhan_tu_truy_van
    assert khong_duoc_co not in nhan_tu_truy_van(cau, bang)


@pytest.mark.parametrize("cau, phai_co", [
    ("trái thơm chín", "Pineapple"),
    ("quả cam vàng", "Orange"),
    ("dàn trống trong lễ hội", "Drum"),
    ("ly nước trên bàn", "Wine glass"),
])
def test_bo_dang_tran_van_khop_duoc_cum_that(bang, cau, phai_co):
    from objects import nhan_tu_truy_van
    assert phai_co in nhan_tu_truy_van(cau, bang)


def test_anh_xa_xap_xi_khi_khong_co_nhan_chinh_xac(bang):
    """Nhãn xấp xỉ là ĐÚNG khi kho không có nhãn chính xác hơn.

    Quy tắc chọn: nếu kho CÓ nhãn chính xác hơn thì phải tách thành dòng riêng
    (`Microwave oven` tồn tại -> "lò vi sóng" không được gộp vào `Oven`). Nếu
    kho KHÔNG có thì ánh xạ về nhãn gần nhất, vì `object_score` cho điểm mềm
    và bỏ trống là mất hẳn.

    Ca `Balloon` có bằng chứng thực nghiệm mạnh: `row_id 7291` trong tập dev
    là khung khinh khí cầu, và detector gán cho nó `Balloon(0,82)` — chính
    OpenImages coi khinh khí cầu là `Balloon`. Kho KHÔNG có `Hot air balloon`.
    Bỏ ánh xạ này là phá đúng một câu trong tập dev của ta.
    """
    from objects import nhan_tu_truy_van
    assert "Balloon" in nhan_tu_truy_van("khinh khí cầu sọc đỏ", bang)
    assert "Microwave oven" in nhan_tu_truy_van("lò vi sóng trong bếp", bang)
    assert "Microwave oven" not in nhan_tu_truy_van("lò nướng bánh", bang)


def test_nhan_bao_trum_that_su_duoc_dung(bang):
    """`row_id 15180` (hai con mèo rừng) được detector gán `Carnivore(0,86)`,
    KHÔNG phải `Cat`. Truy vấn "mèo rừng" phải chạm được nhãn đó."""
    from objects import nhan_tu_truy_van
    assert "Carnivore" in nhan_tu_truy_van("hai con mèo rừng trên bãi cỏ", bang)


def test_cum_chong_lan_giu_ca_hai(bang):
    """"chiếc ô tô": cụm "chiếc ô" và "ô tô" chồng lấn MỘT PHẦN.

    Ăn token tham lam trái-sang-phải thì "chiếc ô" chặn mất "ô tô" và `Car`
    biến mất. Đã đo lỗi thật đó. Chỉ được bỏ cụm nằm HẲN trong cụm khác.
    """
    from objects import nhan_tu_truy_van
    assert "Car" in nhan_tu_truy_van("chiếc ô tô màu đỏ", bang)


def test_cum_dai_thang_cum_ngan(bang):
    from objects import nhan_tu_truy_van
    assert "Human hand" in nhan_tu_truy_van("bàn tay", bang)
    assert "Table" not in nhan_tu_truy_van("bàn tay", bang)
    assert "Table" in nhan_tu_truy_van("cái bàn", bang)


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
