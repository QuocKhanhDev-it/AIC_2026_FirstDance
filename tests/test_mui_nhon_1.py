"""
test_mui_nhon_1.py — Chốt cho đường ống KIS & Q&A của Giai đoạn 2.

Ba thứ được canh kỹ nhất ở đây, vì cả ba đều hỏng ÂM THẦM:

* `uu_tien` **không được làm mất ứng viên nào** — nó là dạng mềm của Bước 1, và
  toàn bộ lý do nó tồn tại là để sai cũng không mất câu.
* `khung_ngu_canh` phải lọc theo GIÂY, không theo số bước — mật độ keyframe
  không đều (A1), lọc nhầm là đưa VLM nhìn sang cảnh khác rồi đếm.
* `gan_dap_an` **không bao giờ được để `answer` rỗng** — `nop_bai.soat` chặn
  cả gói khi gặp, mà chặn là không ghi file nào.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import mui_nhon_1 as M                            # noqa: E402
from schema import Candidate                      # noqa: E402


def master_gia(n_video=3, n_khung=6, buoc_giay=1.0, thu_muc=None) -> pd.DataFrame:
    """Bảng cái tí hon. `thu_muc` khác None thì tạo ảnh giả và điền `kf_path`."""
    d = []
    for v in range(n_video):
        for i in range(n_khung):
            p = None
            if thu_muc is not None:
                f = Path(thu_muc) / f"L01_V{v:03d}_{i}.jpg"
                f.write_bytes(b"\xff\xd8\xff")
                p = str(f)
            d.append({"row_id": len(d), "video_id": f"L01_V{v:03d}",
                      "kf_n": i, "frame_idx": i * 100,
                      "pts_time": float(i) * buoc_giay, "fps": 25.0,
                      "kf_path": p, "title": "", "description": "",
                      "keywords": ""})
    return pd.DataFrame(d)


def uv_gia(cap: list) -> list:
    """`[(row_id, video_id), ...]` -> danh sách ứng viên, điểm giảm dần."""
    return [Candidate(row_id=r, video_id=v, frame_idx=r * 100,
                      score=1.0 - i * 0.01, source="thu")
            for i, (r, v) in enumerate(cap)]


class K2Gia:
    """Kênh 2 giả: điểm tài liệu đặt sẵn, không cần BM25 thật."""

    def __init__(self, video_id, diem):
        self.video_id = list(video_id)
        self._d = np.asarray(diem, dtype=float)

    def diem_tai_lieu(self, cau):
        return self._d


# ---- Bước 1 ---------------------------------------------------------------

def test_video_uu_tien_xep_theo_diem_va_bo_video_khong_khop():
    k2 = K2Gia(["A", "B", "C", "D"], [0.0, 5.0, 1.0, 3.0])
    assert M.video_uu_tien(k2, "câu hỏi", so_video=10) == ["B", "D", "C"]


def test_video_uu_tien_cat_dung_so_video():
    k2 = K2Gia(["A", "B", "C"], [1.0, 5.0, 3.0])
    assert M.video_uu_tien(k2, "câu hỏi", so_video=2) == ["B", "C"]


def test_video_uu_tien_nhan_danh_sach_menh_de_lay_max():
    """Truy vấn dài bị `run.tach_truy_van` cắt thành nhiều mệnh đề (A19/A20).

    Kênh 1 và kênh 2 đều lấy ĐIỂM CAO NHẤT trên từng tài liệu; Bước 1 phải làm
    y hệt, nếu không thì cùng một truy vấn cho ra hai xếp hạng video khác nhau.
    """
    class K2Nhieu(K2Gia):
        def diem_tai_lieu(self, cau):
            return np.array([1.0, 0.0]) if cau == "một" else np.array([0.0, 9.0])

    k2 = K2Nhieu(["A", "B"], [0, 0])
    assert M.video_uu_tien(k2, ["một", "hai"], so_video=5) == ["B", "A"]


def test_be_video_dung_mat_na_tren_bang_cai():
    m = master_gia(n_video=3, n_khung=2)
    be = M.be_video(m, ["L01_V001"])
    assert be.sum() == 2
    assert set(m.video_id[be]) == {"L01_V001"}


def test_uu_tien_khong_lam_mat_ung_vien_nao():
    """Đây là bất biến quan trọng nhất của dạng mềm."""
    uv = uv_gia([(0, "A"), (1, "B"), (2, "C"), (3, "B")])
    ra = M.uu_tien(uv, ["B"])
    assert len(ra) == len(uv)
    assert {c.row_id for c in ra} == {0, 1, 2, 3}


def test_uu_tien_giu_thu_tu_trong_tung_nhom():
    uv = uv_gia([(0, "A"), (1, "B"), (2, "C"), (3, "B")])
    ra = M.uu_tien(uv, ["B"])
    assert [c.row_id for c in ra] == [1, 3, 0, 2]
    assert ra[0].meta["uu_tien"] is True
    assert ra[-1].meta["uu_tien"] is False


def test_uu_tien_khong_sua_danh_sach_goc():
    uv = uv_gia([(0, "A"), (1, "B")])
    M.uu_tien(uv, ["B"])
    assert [c.row_id for c in uv] == [0, 1]
    assert "uu_tien" not in uv[0].meta


def test_thu_hep_bo_that_va_do_la_cho_nguy_hiem():
    uv = uv_gia([(0, "A"), (1, "B"), (2, "C")])
    ra = M.thu_hep(uv, ["B"])
    assert [c.row_id for c in ra] == [1]


def test_thu_hep_video_dung_ngoai_top_N_thi_mat_trang():
    """Ghi lại bằng test cái giá của dạng cứng, để không ai bật nó theo cảm tính."""
    uv = uv_gia([(7, "DUNG"), (1, "B")])
    assert M.thu_hep(uv, ["B", "C"]) == [uv[1]]


# ---- ghép đường ống -------------------------------------------------------

class KenhGia:
    def __init__(self, cap):
        self.cap = cap

    def tim(self, cau, k=100):
        return uv_gia(self.cap)[:k]


def test_truy_hoi_mot_kenh_giu_nguyen_thu_tu():
    kenh = KenhGia([(0, "A"), (1, "B")])
    assert [c.row_id for c in M.truy_hoi("hỏi", [kenh])] == [0, 1]


def test_truy_hoi_hai_kenh_thi_hop_nhat_bang_rrf():
    a = KenhGia([(0, "A"), (1, "B")])
    b = KenhGia([(1, "B"), (2, "C")])
    ra = M.truy_hoi("hỏi", [a, b])
    assert ra[0].row_id == 1                 # ứng viên hai kênh cùng đề cử
    assert ra[0].source == "rrf"


def test_truy_hoi_mac_dinh_khong_bat_buoc_phu_nao():
    """Kỷ luật của repo: không bật gì trước khi thắng trên tập dev."""
    kenh = KenhGia([(0, "A"), (1, "A"), (2, "A")])
    ra = M.truy_hoi("hỏi", [kenh])
    assert len(ra) == 3                      # không dedup, không moi_video


def test_truy_hoi_moi_video_cat_dung_han_muc():
    kenh = KenhGia([(0, "A"), (1, "A"), (2, "B")])
    ra = M.truy_hoi("hỏi", [kenh], moi_video=1)
    assert [c.video_id for c in ra] == ["A", "B"]


def test_truy_hoi_cat_dung_k():
    kenh = KenhGia([(i, "A") for i in range(10)])
    assert len(M.truy_hoi("hỏi", [kenh], k=4)) == 4


def test_truy_hoi_dedup_bo_ban_sao_cung_video():
    """Hai dòng cùng video, vector y hệt -> chỉ giữ dòng điểm cao hơn."""
    mat = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kenh = KenhGia([(0, "A"), (1, "A"), (2, "A")])
    ra = M.truy_hoi("hỏi", [kenh], ma_tran=mat)
    assert [c.row_id for c in ra] == [0, 2]


def test_truy_hoi_buoc_1_mem_can_ca_k2_va_so_video():
    kenh = KenhGia([(0, "A"), (1, "B")])
    k2 = K2Gia(["A", "B"], [0.0, 9.0])
    assert [c.row_id for c in M.truy_hoi("hỏi", [kenh], k2=k2, so_video=1)] == [1, 0]
    # thiếu một trong hai thì Bước 1 im lặng không chạy — đúng ý "mặc định tắt"
    assert [c.row_id for c in M.truy_hoi("hỏi", [kenh], k2=k2)] == [0, 1]


# ---- Bước 4: ngữ cảnh cho VLM ---------------------------------------------

def test_khung_ngu_canh_loc_theo_giay_khong_theo_so_buoc(tmp_path):
    m = master_gia(n_video=1, n_khung=9, buoc_giay=1.5, thu_muc=tmp_path)
    ra = M.khung_ngu_canh(m, row_id=4, so_khung=9, cua_so_giay=2.0)
    assert len(ra) == 3                      # -1,5s / 0 / +1,5s


def test_khung_ngu_canh_dat_khung_chinh_len_dau(tmp_path):
    m = master_gia(n_video=1, n_khung=9, buoc_giay=0.5, thu_muc=tmp_path)
    ra = M.khung_ngu_canh(m, row_id=4, so_khung=3, cua_so_giay=2.0)
    assert ra[0] == m.kf_path.iloc[4]


def test_khung_ngu_canh_khong_vuot_sang_video_khac(tmp_path):
    m = master_gia(n_video=2, n_khung=4, buoc_giay=0.1, thu_muc=tmp_path)
    ra = M.khung_ngu_canh(m, row_id=3, so_khung=9, cua_so_giay=99.0)
    assert all("V000" in p for p in ra)


def test_khung_ngu_canh_bo_qua_dong_khong_co_anh_tren_may_nay():
    m = master_gia(n_video=1, n_khung=4)     # kf_path toàn None (A5.5)
    assert M.khung_ngu_canh(m, row_id=1) == []


# ---- Bước 4: gán đáp án ---------------------------------------------------

def test_gan_dap_an_gan_cung_mot_answer_cho_moi_dong(tmp_path):
    m = master_gia(n_video=2, n_khung=4, thu_muc=tmp_path)
    uv = uv_gia([(0, "L01_V000"), (5, "L01_V001")])
    ra = M.gan_dap_an(uv, m, "mấy cái bát", goi=lambda ch, anh: "5")
    assert [c.meta["answer"] for c in ra] == ["5", "5"]


def test_gan_dap_an_chi_goi_vlm_mot_lan_theo_mac_dinh(tmp_path):
    m = master_gia(n_video=3, n_khung=4, thu_muc=tmp_path)
    uv = uv_gia([(0, "L01_V000"), (4, "L01_V001"), (8, "L01_V002")])
    dem = []
    M.gan_dap_an(uv, m, "hỏi", goi=lambda ch, anh: dem.append(1) or "x")
    assert len(dem) == 1


def test_gan_dap_an_khong_bao_gio_de_answer_rong(tmp_path):
    """Ảnh không có -> vẫn phải ra đáp án, nếu không `nop_bai.soat` chặn cả gói."""
    m = master_gia(n_video=1, n_khung=4)     # không ảnh
    uv = uv_gia([(0, "L01_V000")])
    ra = M.gan_dap_an(uv, m, "hỏi", goi=lambda ch, anh: "không được gọi")
    assert ra[0].meta["answer"] == "không rõ"


def test_gan_dap_an_vlm_tra_chuoi_rong_thi_ve_mac_dinh(tmp_path):
    m = master_gia(n_video=1, n_khung=4, thu_muc=tmp_path)
    uv = uv_gia([(0, "L01_V000")])
    ra = M.gan_dap_an(uv, m, "hỏi", goi=lambda ch, anh: "   ")
    assert ra[0].meta["answer"] == "không rõ"


def test_gan_dap_an_bo_phieu_lay_da_so(tmp_path):
    m = master_gia(n_video=3, n_khung=4, thu_muc=tmp_path)
    uv = uv_gia([(0, "L01_V000"), (4, "L01_V001"), (8, "L01_V002")])
    tra = iter(["5", "3", "3"])
    ra = M.gan_dap_an(uv, m, "hỏi", so_ung_vien=3,
                      goi=lambda ch, anh: next(tra))
    assert ra[0].meta["answer"] == "3"


def test_gan_dap_an_bo_phieu_khong_hoi_lai_cung_video(tmp_path):
    m = master_gia(n_video=2, n_khung=4, thu_muc=tmp_path)
    uv = uv_gia([(0, "L01_V000"), (1, "L01_V000"), (4, "L01_V001")])
    hoi = []
    M.gan_dap_an(uv, m, "hỏi", so_ung_vien=2,
                 goi=lambda ch, anh: hoi.append(anh[0]) or "x")
    assert len(hoi) == 2
    assert "V000" in hoi[0] and "V001" in hoi[1]


def test_gan_dap_an_giu_nguyen_thu_hang_va_khung(tmp_path):
    m = master_gia(n_video=2, n_khung=4, thu_muc=tmp_path)
    uv = uv_gia([(0, "L01_V000"), (5, "L01_V001")])
    ra = M.gan_dap_an(uv, m, "hỏi", goi=lambda ch, anh: "5")
    assert [(c.row_id, c.frame_idx) for c in ra] == [(0, 0), (5, 500)]


def test_gan_dap_an_danh_sach_rong_thi_tra_rong():
    assert M.gan_dap_an([], master_gia(), "hỏi", goi=lambda ch, anh: "x") == []


def test_dap_an_gan_vao_meta_dung_cho_cham_diem_doc(tmp_path):
    """`cham_diem._dung_dap_an` đọc `meta['answer']` — chốt để hai bên không lệch."""
    from cham_diem import _dung_dap_an
    from tap_dev import CauHoi

    m = master_gia(n_video=1, n_khung=4, thu_muc=tmp_path)
    uv = M.gan_dap_an(uv_gia([(0, "L01_V000")]), m, "hỏi",
                      goi=lambda ch, anh: "5")
    hop_le = _dung_dap_an(CauHoi(id="qa-1", loai="QA", cau_hoi="hỏi",
                                 row_id_dung=[0], dap_an="5"))
    assert hop_le(uv[0]) is True


@pytest.mark.parametrize("dap, mong", [("5", True), ("6", False)])
def test_cham_diem_phat_hien_dap_an_sai(tmp_path, dap, mong):
    from cham_diem import _dung_dap_an
    from tap_dev import CauHoi

    m = master_gia(n_video=1, n_khung=4, thu_muc=tmp_path)
    uv = M.gan_dap_an(uv_gia([(0, "L01_V000")]), m, "hỏi",
                      goi=lambda ch, anh: dap)
    hop_le = _dung_dap_an(CauHoi(id="qa-1", loai="QA", cau_hoi="hỏi",
                                 row_id_dung=[0], dap_an="5"))
    assert hop_le(uv[0]) is mong
