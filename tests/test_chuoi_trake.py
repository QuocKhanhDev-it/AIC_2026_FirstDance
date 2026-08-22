"""
test_chuoi_trake.py — Chốt cho việc dóng hàng TRAKE theo thời gian (A39).

Bất biến quan trọng nhất nằm ở `test_mac_dinh_giong_het_ban_cu`: module mới
phải là bản MỞ RỘNG của `run.dong_hang_dp`, không phải bản thay thế. Đặt
`he_so_phat=0` + `trai_toi_da=inf` mà ra kết quả khác bản cũ thì mọi phép đo
sau đó không so được với bất cứ thứ gì — đúng cái bẫy "đổi hai thứ một lúc"
mà kỷ luật đo của dự án cấm.
"""

import math
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from chuoi_trake import (GAN_NHAT_GIAY, TRUNG_VI_TRAI,      # noqa: E402
                         dong_hang_theo_thoi_gian, phat_khoang, rai_deu_hep)
from run import dong_hang_dp                                # noqa: E402


# --------------------------------------------------------------- phat_khoang

def test_lui_ve_qua_khu_la_vo_han():
    """Sự kiện phải TĂNG THẬT — bằng nhau cũng bị chặn, không chỉ lùi."""
    assert phat_khoang(-3.0) == math.inf
    assert phat_khoang(0.0) == math.inf


def test_vung_da_quan_sat_thi_khong_phat():
    """5,1–55,7 s là khoảng đo được; không có cơ sở nói 12 s hơn 28 s."""
    for dt in (5.0, 12.4, 18.7, 28.9, 56.0):
        assert phat_khoang(dt) == 0.0


def test_qua_gan_va_qua_xa_deu_bi_phat_va_phat_muot():
    assert phat_khoang(0.5) > phat_khoang(4.0) > 0.0   # càng gần càng nặng
    assert phat_khoang(200.0) > phat_khoang(60.0) > 0.0  # càng xa càng nặng


# ------------------------------------------------- tương thích với bản cũ

def _bo_thoi_gian(lop):
    """(frame, t, score) -> (frame, score), để gọi được `dong_hang_dp` cũ."""
    return [[(f, s) for f, _, s in ds] for ds in lop]


def test_mac_dinh_giong_het_ban_cu():
    """Không phạt + không trần độ trải => phải trùng `run.dong_hang_dp`.

    Dựng một ca mà thứ tự điểm KHÁC thứ tự thời gian — chính ca từng làm bản
    heuristic cũ hoán đổi khung giữa hai sự kiện.
    """
    lop = [
        [(100, 4.0, 0.9), (900, 36.0, 0.5)],
        [(300, 12.0, 0.4), (80, 3.2, 0.95)],   # ứng viên tốt nhất lại nằm TRƯỚC
        [(700, 28.0, 0.7)],
    ]
    moi = dong_hang_theo_thoi_gian(lop, he_so_phat=0.0, trai_toi_da=math.inf)
    cu = dong_hang_dp(_bo_thoi_gian(lop))
    assert moi == cu


def test_khong_bao_gio_hoan_doi_su_kien():
    """Khung ở vị trí i phải đến TỪ danh sách ứng viên của sự kiện i."""
    lop = [
        [(100, 4.0, 0.9)],
        [(300, 12.0, 0.8)],
        [(700, 28.0, 0.7)],
    ]
    ra = dong_hang_theo_thoi_gian(lop, he_so_phat=1.0, trai_toi_da=180.0)
    assert ra == [100, 300, 700]
    for i, f in enumerate(ra):
        assert f in {x[0] for x in lop[i]}


def test_su_kien_rong_thi_tra_None_chu_khong_do_lech():
    lop = [
        [(100, 4.0, 0.9)],
        [],                       # bóc tách truy vấn hụt một sự kiện
        [(700, 28.0, 0.7)],
    ]
    ra = dong_hang_theo_thoi_gian(lop, he_so_phat=1.0)
    assert ra[0] == 100 and ra[1] is None and ra[2] == 700


def test_tat_ca_rong():
    assert dong_hang_theo_thoi_gian([[], [], []]) == [None, None, None]
    assert dong_hang_theo_thoi_gian([]) == []


# ------------------------------------------------------- prior khoảng cách

def test_prior_bo_chuoi_don_cuc_du_diem_tho_cao_hon():
    """Ba sự kiện cách nhau 0,04 s là vô nghĩa — dù điểm thô của chúng cao hơn.

    Đây đúng dạng `L23_V013,0,1,2,2298` tìm thấy trong bài nộp thật.

    Chốt theo TÍNH CHẤT (không còn cặp nào dưới sàn 5,1 s) chứ không theo một
    bộ số cụ thể: bản đầu của test này chờ `[250,700,1150]`, nhưng DP trả
    `[0,700,1150]` — nó GIỮ được neo điểm cao ở khung 0 mà vẫn đi ra chuỗi
    giãn hợp lý, tức tốt hơn cái test chờ. Prior này phạt DỒN CỤC, nó không
    có nhiệm vụ vứt bỏ một khung điểm cao.
    """
    thoi_gian = {0: 0.0, 1: 0.04, 2: 0.08, 250: 10.0, 700: 28.0, 1150: 46.0}
    lop = [
        [(0, 0.0, 0.90), (250, 10.0, 0.60)],
        [(1, 0.04, 0.90), (700, 28.0, 0.60)],
        [(2, 0.08, 0.90), (1150, 46.0, 0.60)],
    ]
    assert dong_hang_theo_thoi_gian(lop, he_so_phat=0.0) == [0, 1, 2]

    ra = dong_hang_theo_thoi_gian(lop, he_so_phat=1.0)
    t = [thoi_gian[f] for f in ra]
    assert all(b - a >= GAN_NHAT_GIAY for a, b in zip(t, t[1:])), \
        f"vẫn còn cặp dồn cục: {ra} -> {t}"


def test_tran_do_trai_chan_chuoi_vang_khap_video():
    """Chuỗi thật trải trung vị 56,6 s; 40 phút thì không phải một chuỗi."""
    lop = [
        [(100, 4.0, 0.9)],
        [(500, 20.0, 0.9)],
        [(60000, 2400.0, 0.95)],       # cách 40 phút, điểm thô cao nhất
    ]
    assert dong_hang_theo_thoi_gian(lop, trai_toi_da=math.inf)[2] == 60000
    assert dong_hang_theo_thoi_gian(lop, trai_toi_da=180.0)[2] is None


def test_neo_khong_bi_ket_o_ung_vien_diem_cao_nhat():
    """Ứng viên rank-1 của sự kiện 1 có thể là ngõ cụt — phải thử neo khác.

    Ở đây khung 5000 điểm cao nhất nhưng nằm SAU mọi ứng viên còn lại, chọn nó
    thì hai sự kiện sau đều rỗng. Neo 100 điểm thấp hơn nhưng đi trọn chuỗi.
    """
    lop = [
        [(5000, 200.0, 0.99), (100, 4.0, 0.50)],
        [(500, 20.0, 0.50)],
        [(900, 36.0, 0.50)],
    ]
    assert dong_hang_theo_thoi_gian(lop, trai_toi_da=180.0) == [100, 500, 900]


# ----------------------------------------------------------- rai_deu_hep

def test_rai_deu_hep_nam_trong_cua_so_da_do():
    """Không rải khắp video: độ trải phải xấp xỉ TRUNG_VI_TRAI giây."""
    fps = 25.0
    kh = rai_deu_hep(neo_frame=10_000, neo_vi_tri=0, n=3,
                     fps=fps, lo=0, hi=100_000)
    trai_giay = (kh[-1] - kh[0]) / fps
    assert abs(trai_giay - TRUNG_VI_TRAI) < 1.0
    assert kh == sorted(kh) and len(set(kh)) == 3


def test_rai_deu_hep_ton_trong_fps_cua_tung_video():
    """Cùng cửa sổ giây, video 30 fps phải ra khoảng cách KHUNG lớn hơn 25 fps."""
    a = rai_deu_hep(10_000, 0, 3, fps=25.0, lo=0, hi=100_000)
    b = rai_deu_hep(10_000, 0, 3, fps=30.0, lo=0, hi=100_000)
    assert (b[-1] - b[0]) > (a[-1] - a[0])


def test_rai_deu_hep_bi_ken_o_bien_van_tang_that():
    """Kẹp vào [lo,hi] có thể làm hai giá trị bằng nhau — vẫn phải tăng thật."""
    kh = rai_deu_hep(neo_frame=5, neo_vi_tri=2, n=3, fps=25.0, lo=0, hi=10)
    assert kh == sorted(kh) and len(set(kh)) == 3
    assert all(0 <= x for x in kh)


def test_rai_deu_hep_neo_o_giua_thi_co_khung_truoc_neo():
    kh = rai_deu_hep(neo_frame=10_000, neo_vi_tri=1, n=3,
                     fps=25.0, lo=0, hi=100_000)
    assert kh[0] < 10_000 < kh[2]


def test_gan_nhat_giay_khop_voi_so_da_do():
    """Hằng số phải là SỐ ĐO, không phải số chọn tay — min quan sát 5,1 s."""
    assert phat_khoang(5.1) == 0.0
    assert GAN_NHAT_GIAY <= 5.1
