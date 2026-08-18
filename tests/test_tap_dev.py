"""
test_tap_dev.py — Chốt cho tập test giữ kín.

Rò tập test là loại hỏng KHÔNG tự lộ ra: không crash, không sai số, chỉ khiến
con số "kiểm trên tập chưa từng nhìn" ở Giai đoạn 3 trở thành vô nghĩa — mà
lúc đó thì đã muộn để sửa.
"""

import sys
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

DEV = GOC / "dev" / "tap_dev.jsonl"
TEST = GOC / "dev" / "tap_test.jsonl"

pytestmark = pytest.mark.skipif(
    not DEV.exists(), reason="chưa có dev/tap_dev.jsonl")


@pytest.fixture(scope="module")
def master():
    import pandas as pd
    p = GOC / "index" / "master.parquet"
    if not p.exists():
        pytest.skip("chưa có index/master.parquet")
    return pd.read_parquet(p)


def test_hai_tap_khong_giao_nhau():
    """Một câu nằm ở cả hai tập là rò trực tiếp."""
    import tap_dev
    if not TEST.exists():
        pytest.skip("chưa tách tập test")
    d = {c.id for c in tap_dev.doc(DEV)}
    t = {c.id for c in tap_dev.doc(TEST)}
    assert d & t == set(), f"{len(d & t)} câu nằm ở CẢ HAI tập: {sorted(d & t)[:5]}"


def test_gop_khong_nuot_tap_test(tmp_path):
    """`--gop` chạy lại từ file thành viên KHÔNG được kéo câu test về dev.

    Đây là đường rò thật nhất: mỗi lần có người gửi câu mới, ai đó chạy `--gop`,
    và nếu không có chốt thì tập test tan biến trong im lặng.
    """
    import tap_dev
    if not TEST.exists():
        pytest.skip("chưa tách tập test")
    thu_muc = GOC / "dev" / "tap_dev_thanh_vien"
    if not thu_muc.exists():
        pytest.skip("chưa có thư mục file thành viên")

    gop, loi = tap_dev.gop([thu_muc])
    assert not loi, loi
    t = {c.id for c in tap_dev.doc(TEST)}
    lot = {c.id for c in gop} & t
    assert lot == set(), f"{len(lot)} câu test lọt về tập dev sau khi gộp"


def test_tach_test_phan_tang_va_tat_dinh(master):
    """Chia phải PHÂN TẦNG và TẤT ĐỊNH.

    Phân tầng: chia ngẫu nhiên thuần có thể ra tập test không có câu QA nào,
    hoặc dồn hết vào một nhóm L — rồi con số kiểm cuối chẳng đại diện cho gì.

    Tất định: mọi máy, mọi lần chạy phải ra cùng một tập, nếu không thì hai
    người sẽ giữ kín hai tập khác nhau.
    """
    import tap_dev
    cau = tap_dev.doc(DEV) + (tap_dev.doc(TEST) if TEST.exists() else [])
    if len(cau) < 20:
        pytest.skip("cần ≥ 20 câu để kiểm phân tầng")

    d1, t1 = tap_dev.tach_test(cau, master, 0.15)
    d2, t2 = tap_dev.tach_test(cau, master, 0.15)
    assert [c.id for c in t1] == [c.id for c in t2], "chia KHÔNG tất định"
    assert [c.id for c in d1] == [c.id for c in d2]

    assert set(c.id for c in d1) & set(c.id for c in t1) == set()
    assert len(d1) + len(t1) == len(cau), "chia làm mất hoặc nhân đôi câu"

    nhom_dev = {c.nhom(master) for c in d1}
    nhom_test = {c.nhom(master) for c in t1}
    thieu = nhom_dev - nhom_test
    assert len(thieu) <= 1, f"tập test thiếu hẳn các nhóm L: {sorted(thieu)}"


def test_tap_test_van_hop_le():
    """Tập test cũng phải qua đúng bộ soát như tập dev."""
    import tap_dev
    if not TEST.exists():
        pytest.skip("chưa tách tập test")
    loi = tap_dev.kiem(tap_dev.doc(TEST))
    assert not loi, loi


def test_gop_bao_trung_id_KE_CA_khi_id_do_nam_trong_tap_test(tmp_path):
    """Soát trùng `id` phải chạy TRƯỚC bước lọc tập test.

    Làm ngược lại thì một `id` bị nhân đôi mà tình cờ nằm trong tập test sẽ bị
    bỏ qua cả hai lần và không bao giờ được báo. Đã vấp thật khi chuyển một câu
    sang file khác mà quên xoá bản cũ.
    """
    import tap_dev
    if not TEST.exists():
        pytest.skip("chưa tách tập test")
    id_test = tap_dev.doc(TEST)[0].id
    mau = tap_dev.doc(TEST)[:1]
    for ten in ("a.jsonl", "b.jsonl"):
        tap_dev.ghi(mau, tmp_path / ten)

    _, loi = tap_dev.gop([tmp_path])
    assert any(id_test in x for x in loi), (
        f"id '{id_test}' bị nhân đôi mà `gop()` không báo: {loi}")


def test_id_ghi_sai_nhom_thi_bi_bat(master):
    """`id` ghi nhóm L nào thì đáp án phải nằm ở nhóm đó.

    Đặt nhầm file là lỗi KHÔNG AI KIỂM ĐƯỢC: người giữ nhóm ghi trên `id`
    không có ảnh để mở, người giữ nhóm thật thì không biết câu đó tồn tại.
    """
    import tap_dev
    r = 500
    dung_nhom = master.video_id.iloc[r][:3]
    sai = "L99" if dung_nhom != "L99" else "L98"
    c = tap_dev.CauHoi(id=f"kis-{sai}-001", loai="KIS", cau_hoi="x",
                       row_id_dung=[r])
    loi = tap_dev.kiem([c])
    assert any("đặt nhầm file" in x for x in loi), loi
