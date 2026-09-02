"""Kiểm `hop_diem.py` — hợp nhất kênh bằng điểm đã chuẩn hoá."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hop_diem import chuan_hoa, hop_nhat_diem       # noqa: E402
from schema import Candidate                        # noqa: E402


def uv(*diem, nguon="k"):
    """Danh sách ứng viên giả, row_id 0,1,2… theo đúng thứ tự truyền vào."""
    return [Candidate(row_id=i, video_id=f"V{i}", frame_idx=i * 10,
                      score=float(d), source=nguon)
            for i, d in enumerate(diem)]


# ---------------------------------------------------------------- chuan_hoa

def test_rong_tra_ve_rong():
    assert chuan_hoa([]) == {}


def test_moi_diem_bang_nhau_khong_chia_cho_khong():
    """Bẫy thật: sd = 0 và hi - lo = 0 -> nan hoặc ZeroDivisionError.

    Tính chất cần giữ KHÔNG phải "bằng 0" mà là "bằng NHAU": một kênh không
    phân biệt nổi ứng viên nào thì nó đóng góp một HẰNG SỐ, và hằng số không
    đổi thứ hạng. `sigmoid` cho 0,5 chứ không phải 0 (vì σ(0) = 0,5) — vẫn
    đúng tính chất đó.
    """
    for cach in ("z", "minmax", "sigmoid"):
        v = list(chuan_hoa(uv(5, 5, 5), cach=cach).values())
        assert len(set(v)) == 1, cach
        assert all(x == x for x in v), f"{cach} ra nan"


def test_z_dung_cong_thuc():
    v = chuan_hoa(uv(1, 2, 3), cach="z")
    assert v[0] == pytest.approx(-1.2247, abs=1e-4)
    assert v[1] == pytest.approx(0.0)
    assert v[2] == pytest.approx(1.2247, abs=1e-4)


def test_minmax_ve_khoang_0_1():
    v = chuan_hoa(uv(2, 4, 10), cach="minmax")
    assert v[0] == pytest.approx(0.0)
    assert v[2] == pytest.approx(1.0)
    assert 0.0 < v[1] < 1.0


def test_sigmoid_giu_nguyen_thu_tu():
    """σ đơn điệu -> KHÔNG đổi thứ hạng nội bộ kênh. Đây là điểm chính của A?"""
    tho = [3.0, 1.0, 7.0, 5.0]
    for tau in (0.1, 1.0, 20.0):
        v = chuan_hoa(uv(*tho), cach="sigmoid", tau=tau)
        assert sorted(v, key=lambda r: -v[r]) == [2, 3, 0, 1], tau


def test_tau_lon_lam_nhon_khoang_cach():
    """tau lớn -> gần bậc thang: ứng viên đầu tách hẳn khỏi phần còn lại."""
    it = chuan_hoa(uv(1, 2, 9), cach="sigmoid", tau=0.2)
    nhieu = chuan_hoa(uv(1, 2, 9), cach="sigmoid", tau=8.0)
    assert (nhieu[2] - nhieu[1]) > (it[2] - it[1])


def test_log1p_nan_duoi():
    """BM25 lệch: một giá trị vọt cao kéo z-score của phần còn lại xuống.

    ⚠️ Phân phối chỉ có HAI giá trị (kiểu 0,0,0,100) thì z-score do SỐ LƯỢNG
    quyết định chứ không do độ lớn, nên mọi phép nắn đơn điệu cho kết quả y
    hệt. Muốn thấy tác dụng của log1p phải có ít nhất ba giá trị khác nhau.
    """
    tho = uv(0, 1, 2, 100)
    khong_nan = chuan_hoa(tho, cach="z")
    co_nan = chuan_hoa(tho, cach="z", truoc="log1p")
    assert co_nan[3] < khong_nan[3]

    hai_gia_tri = uv(0, 0, 0, 100)
    assert (chuan_hoa(hai_gia_tri, cach="z")
            == chuan_hoa(hai_gia_tri, cach="z", truoc="log1p"))


def test_cach_va_truoc_sai_thi_bao_loi():
    with pytest.raises(ValueError):
        chuan_hoa(uv(1, 2), cach="softmax")
    with pytest.raises(ValueError):
        chuan_hoa(uv(1, 2), truoc="sqrt")


# ------------------------------------------------------------ hop_nhat_diem

def test_mot_kenh_giu_nguyen_thu_tu():
    ds = uv(9, 5, 1)
    ra = hop_nhat_diem([ds])
    assert [c.row_id for c in ra] == [0, 1, 2]


def test_ung_vien_vang_mat_khong_bi_chon_xuong_day():
    """A60 từng bù -1e9 và chôn mọi ứng viên chưa chấm. Bù = MIN của kênh đó.

    `r0` chỉ có ở kênh 1 nhưng là ứng viên MẠNH NHẤT ở đó; `r9` có ở cả hai
    kênh nhưng kém ở cả hai. r0 phải đứng trên r9.
    """
    k1 = [Candidate(0, "V0", 0, 10.0), Candidate(9, "V9", 90, 1.0)]
    k2 = [Candidate(9, "V9", 90, 1.0), Candidate(8, "V8", 80, 0.5)]
    ra = hop_nhat_diem([k1, k2], cach="z")
    hang = [c.row_id for c in ra]
    assert hang.index(0) < hang.index(9)


def test_trong_so_0_lam_kenh_do_bien_mat_khoi_thu_hang():
    k1 = uv(9, 5, 1)
    k2 = [Candidate(2, "V2", 20, 100.0), Candidate(1, "V1", 10, 50.0)]
    a = hop_nhat_diem([k1, k2], trong_so=[1.0, 0.0])
    b = hop_nhat_diem([k1])
    assert [c.row_id for c in a] == [c.row_id for c in b]


def test_hop_nhat_bao_loi_khi_lech_so_luong():
    with pytest.raises(ValueError):
        hop_nhat_diem([uv(1, 2)], trong_so=[1.0, 1.0])
    with pytest.raises(ValueError):
        hop_nhat_diem([uv(1, 2)], truoc=[None, "log1p"])
    with pytest.raises(ValueError):
        hop_nhat_diem([uv(1, 2)], bu="trung binh")


def test_bu_zero_khac_bu_min():
    k1 = [Candidate(0, "V0", 0, 10.0), Candidate(1, "V1", 10, 9.0)]
    k2 = [Candidate(1, "V1", 10, 5.0), Candidate(2, "V2", 20, 1.0)]
    m = {c.row_id: c.score for c in hop_nhat_diem([k1, k2], cach="z", bu="min")}
    z = {c.row_id: c.score for c in hop_nhat_diem([k1, k2], cach="z", bu="zero")}
    assert m[0] != z[0]


def test_giu_frame_idx_va_video_id_cua_ung_vien_goc():
    """frame_idx là giá trị NỘP — không được sinh lại hay để lẫn."""
    k1 = [Candidate(7, "L21_V003", 26533, 0.9)]
    ra = hop_nhat_diem([k1])
    assert ra[0].frame_idx == 26533 and ra[0].video_id == "L21_V003"


def test_log1p_thuc_su_di_toi_hop_nhat():
    """Bảo vệ đường dây `truoc` — A?: dòng z-score và z-score+log1p ra ĐIỂM Y
    HỆT nhau tới 4 chữ số, nên phải chắc đó là kết luận thật chứ không phải
    tham số bị nuốt trên đường truyền.

    Dựng phân phối lệch kiểu BM25 (phần lớn gần 0, một nhúm vọt cao).
    """
    k1 = uv(0.9, 0.8, 0.7)
    k3 = [Candidate(10 + i, f"W{i}", i, s)
          for i, s in enumerate([40.0, 3.0, 2.0, 1.0, 0.5, 0.1])]
    khong = hop_nhat_diem([k1, k3], trong_so=[1.0, 0.5], cach="z")
    co = hop_nhat_diem([k1, k3], trong_so=[1.0, 0.5], cach="z",
                       truoc=[None, "log1p"])
    assert [c.score for c in khong] != [c.score for c in co]
