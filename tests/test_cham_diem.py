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


# ---- chấm TRAKE theo mô hình BÀI NỘP --------------------------------------
#
# `diem_trake()` chấm KÊNH: một danh sách ứng viên, mỗi sự kiện có nằm trong đó
# không. `diem_trake_bai_nop()` chấm BÀI NỘP: mỗi dòng là một BỘ N khung, vị trí
# i chỉ được so với sự kiện i. Kênh tìm đủ mà lắp sai vị trí -> BTC cho 0.

def test_trake_bai_nop_dung_het_thi_tron_diem():
    from cham_diem import diem_trake_bai_nop
    assert diem_trake_bai_nop([[1, 2, 3]], [{1}, {2}, {3}]) == pytest.approx(1.0)


def test_trake_bai_nop_cham_tung_phan():
    """A8.1: *"partial credit proportional to the number of correctly matched
    frames"* — đúng 2/3 vị trí thì được 2/3."""
    from cham_diem import diem_trake_bai_nop
    assert diem_trake_bai_nop([[1, 2, 9]], [{1}, {2}, {3}]) == pytest.approx(2 / 3)


def test_trake_bai_nop_SAI_VI_TRI_thi_khong_duoc_diem():
    """Chốt cho khác biệt quan trọng nhất: đủ cả ba khung đúng nhưng LẮP SAI
    THỨ TỰ thì BTC cho 0. `diem_trake()` không bắt được điều này."""
    from cham_diem import diem_trake_bai_nop
    assert diem_trake_bai_nop([[3, 2, 1]], [{1}, {2}, {3}]) == pytest.approx(1 / 3)


def test_trake_bai_nop_lay_dong_TOT_NHAT_trong_top_k():
    """`R@k = max R-Score trong top-k`. Dòng đúng ở hạng 2 vẫn kéo được R@5."""
    from cham_diem import diem_trake_bai_nop
    d = diem_trake_bai_nop([[9, 9, 9], [1, 2, 3]], [{1}, {2}, {3}])
    assert d == pytest.approx((0 + 1 + 1 + 1 + 1) / 5)


def test_trake_bai_nop_rong_thi_khong_no_loi():
    from cham_diem import diem_trake_bai_nop
    assert diem_trake_bai_nop([], [{1}, {2}]) == 0.0
    assert diem_trake_bai_nop([[1, 2]], []) == 0.0


# ── tang NOP cua TRAKE: nhan ca row_id lan set row_id ─────────────────

def test_diem_trake_bai_nop_nhan_ca_row_id_lan_set():
    """A5.7: 614 keyframe dung chung frame_idx, nen mot vi tri co the la SET.

    `run.dung_trake()` tra frame_idx -> tra ra nhieu row_id; `78_`/`91_` tra
    thang row_id. Mot ban phai lo ca hai, khong thi hai script tu viet hai ban
    va chung se troi khoi nhau.
    """
    from cham_diem import diem_trake_bai_nop
    dung = [{10, 11}, {20}]
    assert diem_trake_bai_nop([[10, 20]], dung) == 1.0        # row_id tran
    assert diem_trake_bai_nop([[{11}, {20}]], dung) == 1.0     # set
    assert diem_trake_bai_nop([[{99, 11}, {20}]], dung) == 1.0  # set co lan
    assert diem_trake_bai_nop([[10, 99]], dung) == 0.5         # sai mot vi tri
    assert diem_trake_bai_nop([[20, 10]], dung) == 0.0         # dung khung,
    #                                                            SAI vi tri


def test_diem_trake_bai_nop_cham_theo_VI_TRI_chu_khong_theo_tap():
    """Cho phan biet tang NOP voi tang KENH — day la ca A63 do duoc mat 52%."""
    from cham_diem import diem_trake, diem_trake_bai_nop
    dung = [{1}, {2}, {3}]
    # Kenh tim ra du ba su kien, nhung xep nguoc thu tu.
    assert diem_trake_bai_nop([[3, 2, 1]], dung) < 1.0
    # Tang KENH thi khong thay van de do: no nhan HANG cua tung su kien, va
    # ca ba su kien deu tim thay o hang 1.
    assert diem_trake([1, 1, 1]) == 1.0


def test_bao_cao_tu_bang_in_nguong_nhieu_va_ket_luan():
    """Phan bao cao dung lai duoc cho bang dung san — khong qua `cham()`."""
    import pandas as pd
    from cham_diem import bao_cao_tu_bang

    def bang_diem(diem):
        return pd.DataFrame({"id": [f"c{i}" for i in range(len(diem))],
                             "loai": ["TRAKE"] * len(diem), "diem": diem})

    bang = {
        "moc": {2.0: bang_diem([0.0] * 10), 15.0: bang_diem([0.0] * 10)},
        "tot hon": {2.0: bang_diem([1.0] * 10), 15.0: bang_diem([1.0] * 10)},
    }
    ra = bao_cao_tu_bang(bang, moc=(2.0, 15.0))
    assert "10 câu | mốc nền: moc" in ra
    assert "ngưỡng" in ra                      # co in nguong nhieu
    assert "10-0-0" in ra                      # thang-thua-hoa
    assert "ON DINH" in ra


# ── phan biet hai TANG cham diem TRAKE ────────────────────────────────

def test_la_bai_nop_trake_phan_biet_dung():
    from cham_diem import la_bai_nop_trake
    from schema import Candidate
    assert la_bai_nop_trake([[1, 2, 3], [4, 5, 6]])          # dong bai nop
    assert la_bai_nop_trake([({1}, {2})])                     # dong dang set
    assert not la_bai_nop_trake([Candidate(1, "V0", 0, 0.9)])  # ung vien
    assert not la_bai_nop_trake([])


def test_cham_TRAKE_tu_nhan_ra_tang_NOP_khi_duoc_dua_dong():
    """Ham cau hinh tra ve DONG DA LAP -> cham tang NOP, khong phai tang KENH.

    Day la ca A63: kenh tim ra du su kien ma xep sai vi tri thi tang KENH cho
    diem cao con BTC cho 0.
    """
    import pandas as pd
    from cham_diem import cham
    from schema import Candidate
    from tap_dev import CauHoi

    c = CauHoi(id="t1", loai="TRAKE", cau_hoi="x", row_id_dung=[[0], [1]])
    m = pd.DataFrame({"row_id": [0, 1], "video_id": ["V0", "V0"],
                      "pts_time": [0.0, 10.0], "frame_idx": [0, 250]})

    dung = cham([c], lambda _: [[0, 1]], master=m, dung_sai_giay=None)
    assert dung.tang.iloc[0] == "nộp" and dung.diem.iloc[0] == 1.0

    nguoc = cham([c], lambda _: [[1, 0]], master=m, dung_sai_giay=None)
    assert nguoc.tang.iloc[0] == "nộp" and nguoc.diem.iloc[0] == 0.0

    # Cung bo ung vien do, cham o tang KENH thi KHONG thay van de xep nguoc:
    # 0,9 (su kien 2 o hang 2 nen mat R@1) — trong khi bai nop xep nguoc la 0,0.
    # Do la khoang cach A63 do duoc, va la ly do phai phan biet hai tang.
    kenh = cham([c], lambda _: [Candidate(0, "V0", 0, 0.9),
                                Candidate(1, "V0", 250, 0.8)],
                master=m, dung_sai_giay=None)
    assert kenh.tang.iloc[0] == "kênh"
    assert kenh.diem.iloc[0] == 0.9 > nguoc.diem.iloc[0]


def test_bao_cao_keu_to_khi_TRAKE_cham_o_tang_KENH():
    """Sai lech AM THAM phai thanh sai lech NHIN THAY DUOC."""
    import pandas as pd
    from cham_diem import bao_cao_tu_bang

    def bang_diem(tang):
        return pd.DataFrame({"id": ["a", "b"], "loai": ["TRAKE", "KIS"],
                             "tang": tang, "diem": [0.5, 0.5]})

    co = {"moc": {2.0: bang_diem(["kênh", "nộp"]),
                  15.0: bang_diem(["kênh", "nộp"])}}
    assert "tầng KÊNH" in bao_cao_tu_bang(co, moc=(2.0, 15.0))

    khong = {"moc": {2.0: bang_diem(["nộp", "nộp"]),
                     15.0: bang_diem(["nộp", "nộp"])}}
    assert "tầng KÊNH" not in bao_cao_tu_bang(khong, moc=(2.0, 15.0))
