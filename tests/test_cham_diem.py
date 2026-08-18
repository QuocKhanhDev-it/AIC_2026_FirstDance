"""
test_cham_diem.py — Chốt cho thước đo.

Thước đo sai thì MỌI kết luận sau đó sai theo mà không có gì báo. Đây là loại
lỗi tệ nhất trong dự án: nó không làm crash, không làm test đỏ, chỉ lặng lẽ
xếp hạng các cấu hình theo thứ tự sai.
"""

import sys
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

pytestmark = pytest.mark.skipif(
    not (GOC / "index" / "master.parquet").exists(),
    reason="chưa có index/master.parquet")


@pytest.fixture(scope="module")
def master():
    import pandas as pd
    return pd.read_parquet(GOC / "index" / "master.parquet")


def test_bac_thang_dung_cong_thuc_btc():
    """`Final Score = trung bình R@{1,5,20,50,100}` (PHẦN C)."""
    from cham_diem import diem_cau
    assert diem_cau(1) == pytest.approx(1.0)
    assert diem_cau(2) == pytest.approx(0.8)
    assert diem_cau(5) == pytest.approx(0.8)
    assert diem_cau(6) == pytest.approx(0.6)
    assert diem_cau(20) == pytest.approx(0.6)
    assert diem_cau(50) == pytest.approx(0.4)
    assert diem_cau(100) == pytest.approx(0.2)
    assert diem_cau(101) == pytest.approx(0.0)
    assert diem_cau(None) == pytest.approx(0.0)


def test_trake_cham_tung_phan():
    """A8.1 + A9: TRAKE chấm TỪNG PHẦN theo số sự kiện khớp.

    Nên bỏ trống một vị trí chỉ mất phần điểm đó, KHÔNG mất cả câu — đó là lý
    do PHẦN C mục 3 bảo luôn điền đủ N.
    """
    from cham_diem import diem_trake
    assert diem_trake([1, 1, 1]) == pytest.approx(1.0)
    assert diem_trake([1, None, 1]) == pytest.approx(2 / 3)
    assert diem_trake([None, None, None]) == pytest.approx(0.0)
    assert 0 < diem_trake([1, None, None]) < 1


def test_no_cua_so_chi_lay_cung_video(master):
    """Cửa sổ là khoảng thời gian TRONG MỘT video — không được tràn sang video
    khác dù `pts_time` có gần nhau."""
    from cham_diem import no_cua_so
    r = 500
    ra = no_cua_so([r], master, 30.0)
    vid = master.video_id.iloc[r]
    assert all(master.video_id.iloc[x] == vid for x in ra)


def test_no_cua_so_no_theo_dung_sai(master):
    """Dung sai càng rộng thì càng nhiều keyframe được tính đúng.

    Đo thật trên tập dev: trung vị 1 keyframe (chính xác) -> 2 (±2s) -> 3
    (±5s) -> 10 (±15s) -> 80 (±150s).
    """
    from cham_diem import no_cua_so
    r = 500
    kich_co = [len(no_cua_so([r], master, ds)) for ds in (0.5, 2, 5, 30)]
    assert kich_co == sorted(kich_co), "nới dung sai mà tập đáp án không lớn lên"
    assert kich_co[0] >= 1


def test_cham_chat_hon_btc_neu_quen_dung_sai(master):
    """Chốt cho lỗi nguy hiểm nhất: chấm CHẶT HƠN BTC.

    BTC chấp nhận cả cửa sổ 4 giây–5 phút (A9). Nếu ta so `row_id` chính xác
    thì một keyframe cách đáp án 2 giây bị tính SAI, trong khi BTC tính ĐÚNG.
    Điểm đo được thấp hơn thật, và thấp KHÔNG ĐỀU giữa các cấu hình.
    """
    from cham_diem import cham
    from schema import Candidate
    from tap_dev import CauHoi

    r = 500
    g = master.iloc[r]
    ke = master[(master.video_id == g.video_id)
                & ((master.pts_time - g.pts_time).abs() <= 3.0)
                & (master.row_id != r)]
    if ke.empty:
        pytest.skip("row 500 không có keyframe lân cận trong 3 giây")
    lan_can = int(ke.row_id.iloc[0])

    cau = [CauHoi(id="t1", loai="KIS", cau_hoi="x", row_id_dung=[r])]

    def tra_ve_lan_can(c):
        x = master.iloc[lan_can]
        return [Candidate(lan_can, x.video_id, int(x.frame_idx), 1.0, "test")]

    chat = cham(cau, tra_ve_lan_can)
    assert chat.diem.iloc[0] == 0.0, "chấm chính xác phải coi lân cận là SAI"

    rong = cham(cau, tra_ve_lan_can, master=master, dung_sai_giay=5.0)
    assert rong.diem.iloc[0] == 1.0, "chấm theo cửa sổ phải coi lân cận là ĐÚNG"


def test_dung_sai_doi_master(master):
    from cham_diem import cham
    with pytest.raises(ValueError):
        cham([], lambda c: [], dung_sai_giay=5.0)


# ---- câu Q&A: `answer` phải xét THEO TỪNG ỨNG VIÊN ------------------------
#
# Bài nộp Q&A là một danh sách xếp hạng các bộ (video_id, frame_idx, answer) —
# MỖI DÒNG mang `answer` riêng. Một dòng ăn điểm khi khung nằm trong cửa sổ VÀ
# chuỗi `answer` đúng. Nên thứ hạng phải là dòng ĐẦU TIÊN thỏa CẢ HAI.

def _cau_qa(r_dung, dap_an="3"):
    from tap_dev import CauHoi
    return CauHoi(id="qa1", loai="QA", cau_hoi="mấy người",
                  row_id_dung=[r_dung], dap_an=dap_an)


def _kq(master, cac_bo):
    """[(row_id, answer), ...] -> list[Candidate] theo đúng thứ tự."""
    from schema import Candidate
    ra = []
    for r, tra in cac_bo:
        x = master.iloc[r]
        ra.append(Candidate(r, x.video_id, int(x.frame_idx), 1.0, "test",
                            meta={"answer": tra}))
    return ra


def test_qa_dap_an_sai_o_hang_1_khong_duoc_giet_ca_cau(master):
    """Hạng 1 sai đáp án, hạng 4 đúng cả khung lẫn đáp án -> phải được 0,8.

    Chấm 0 ở đây là **hạ điểm oan mọi cấu hình biết trả lời đúng ở hạng sau**,
    và hạ không đều — đúng loại lệch làm đảo thứ hạng giữa các cấu hình.
    """
    from cham_diem import cham
    r = 500
    khac = int(master.row_id.iloc[r + 50])
    kq = _kq(master, [(khac, "9"), (khac, "9"), (khac, "9"), (r, "3")])
    d = cham([_cau_qa(r)], lambda c: kq)
    assert d.diem.iloc[0] == pytest.approx(0.8), (
        "đáp án của hạng 1 không được quyết định cả câu")


def test_qa_dap_an_dung_o_hang_1_khong_cuu_duoc_khung_sai(master):
    """Ngược lại: hạng 1 đúng đáp án nhưng SAI khung, hạng 4 đúng khung nhưng
    SAI đáp án -> phải được 0. Không dòng nào thỏa cả hai điều kiện."""
    from cham_diem import cham
    r = 500
    khac = int(master.row_id.iloc[r + 50])
    kq = _kq(master, [(khac, "3"), (khac, "3"), (khac, "3"), (r, "9")])
    d = cham([_cau_qa(r)], lambda c: kq)
    assert d.diem.iloc[0] == pytest.approx(0.0), (
        "đáp án đúng ở một khung SAI không được cứu câu")


def test_qa_kenh_khong_sinh_answer_thi_chi_cham_truy_hoi(master):
    """Kênh chưa biết trả lời (mọi kênh hiện tại) không bị phạt.

    Giữ nguyên hành vi cũ có chủ ý: lúc này ta đang đo TRUY HỒI, chưa đo khả
    năng trả lời. Đổi chỗ này là làm mọi con số cũ hết so được.
    """
    from cham_diem import cham
    from schema import Candidate
    r = 500
    x = master.iloc[r]
    kq = [Candidate(r, x.video_id, int(x.frame_idx), 1.0, "test")]
    assert cham([_cau_qa(r)], lambda c: kq).diem.iloc[0] == pytest.approx(1.0)
