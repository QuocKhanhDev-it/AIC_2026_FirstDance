"""
test_run.py — Chốt cho đường ống đầu-cuối.

Phần KHÔNG cần model: đọc đề, tách sự kiện TRAKE, dựng dòng TRAKE. Ba thứ này
mới là chỗ mất trắng cả câu nếu sai — tách nhầm số sự kiện là sai định dạng,
BTC không chấm.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                   # noqa: E402
from schema import Candidate                      # noqa: E402


def master_gia(n_video=3, n_khung=20) -> pd.DataFrame:
    d = []
    for v in range(n_video):
        for i in range(n_khung):
            d.append({"row_id": len(d), "video_id": f"L01_V{v:03d}",
                      "kf_n": i, "frame_idx": i * 500, "pts_time": float(i * 20),
                      "fps": 25.0, "kf_name": None, "kf_path": None,
                      "title": "", "description": "", "keywords": ""})
    return pd.DataFrame(d)


def uv(video, frames):
    return [Candidate(row_id=0, video_id=video, frame_idx=f, score=1.0 - i * 0.01)
            for i, f in enumerate(frames)]


# ---- đọc đề ---------------------------------------------------------------

def test_doc_de_theo_hau_to(tmp_path):
    for t in ("query-1-kis", "query-2-qa", "query-4-trake"):
        (tmp_path / f"{t}.txt").write_text("nội dung", encoding="utf-8")
    (tmp_path / "ghi_chu.txt").write_text("bỏ qua", encoding="utf-8")
    de = R.doc_de(tmp_path)
    assert set(de) == {"query-1-kis", "query-2-qa", "query-4-trake"}
    assert R.loai_cua("query-4-trake") == "trake"


def test_thu_muc_de_rong_thi_dung_ngay(tmp_path):
    with pytest.raises(SystemExit):
        R.doc_de(tmp_path)


# ---- tách sự kiện TRAKE ---------------------------------------------------

def test_tach_su_kien_theo_dong_va_bo_so_thu_tu():
    sk = R.tach_su_kien("1. người bước vào\n2. người ngồi xuống\n3. người đứng lên")
    assert sk == ["người bước vào", "người ngồi xuống", "người đứng lên"]


def test_tach_su_kien_mot_dong_co_danh_so():
    assert len(R.tach_su_kien("1) mở cửa 2) bước vào 3) đóng cửa")) == 3


def test_tach_su_kien_theo_dau_cham_phay():
    assert len(R.tach_su_kien("mở cửa; bước vào; đóng cửa")) == 3


def test_tach_su_kien_mot_su_kien_van_ra_mot():
    assert R.tach_su_kien("một người đàn ông đi bộ") == ["một người đàn ông đi bộ"]


def test_tach_su_kien_moc_E_khong_can_dau_hai_cham():
    """Đề MẪU viết `E1:`, đề THẬT viết `E1 ` — cả hai phải ra cùng số sự kiện.

    Regex bản đầu đòi `[:.]` bắt buộc nên không khớp dòng nào của đề thật, rơi
    xuống nhánh mỗi-dòng-một-sự-kiện và biến LỜI MỞ ĐẦU thành sự kiện 1: nộp 4
    Frame ID nơi BTC đòi 3. Sai số Frame ID là sai định dạng, mất trắng cả gói.
    """
    that = ("Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng.\n"
            "E1 Khoảnh khắc đầu tiên xuất hiện hai con rồng vàng.\n"
            "E2 Khoảnh khắc đầu tiên lân hoàn tất cú xoay người.\n"
            "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng.")
    mau = that.replace("E1 ", "E1: ").replace("E2 ", "E2: ").replace("E3 ", "E3: ")
    assert len(R.tach_su_kien(that)) == 3
    assert len(R.tach_su_kien(mau)) == 3
    # Lời mở đầu là BỐI CẢNH CHUNG, ghép vào mọi sự kiện, không phải sự kiện.
    assert all("con lân trắng" in s for s in R.tach_su_kien(that))
    assert "E1" not in R.tach_su_kien(that)[0]


def test_tach_su_kien_khong_nhan_nham_E_lien_chu():
    """Vẫn đòi một dấu tách sau số, để `E12abc` không thành sự kiện 12."""
    assert R.tach_su_kien("E12abc một dòng") == ["E12abc một dòng"]


def test_tach_su_kien_dem_theo_dong_khong_theo_so():
    """Đề mẫu `query-p1-18-trake` đánh nhầm E1,E2,E2,E4 — bốn dòng là bốn sự kiện."""
    assert len(R.tach_su_kien("E1: a\nE2: b\nE2: c\nE4: d")) == 4


# ---- dựng dòng TRAKE ------------------------------------------------------

def test_trake_tang_dan_theo_thoi_gian():
    """BTC: *'thứ tự phải tuân theo thứ tự thời gian của các events'*."""
    m = master_gia()
    ds = R.dung_trake([uv("L01_V000", [5000]), uv("L01_V000", [1000]),
                       uv("L01_V000", [3000])], m)
    assert ds, "không dựng được dòng nào"
    f = ds[0].frame_idxs
    assert f == sorted(f) and len(set(f)) == len(f), f


def test_trake_du_N_khung_ke_ca_khi_thieu_su_kien():
    """TRAKE chấm TỪNG PHẦN — bỏ trống chắc chắn 0, đoán sai cũng 0."""
    m = master_gia()
    ds = R.dung_trake([uv("L01_V000", [1000]), [], uv("L01_V000", [7000])], m)
    assert ds and len(ds[0].frame_idxs) == 3


def test_trake_noi_suy_vao_GIUA_hai_neo():
    """Chỗ trống ở giữa phải nội suy, không phải nhét sát mép."""
    m = master_gia()
    f = R.dung_trake([uv("L01_V000", [1000]), [], uv("L01_V000", [9000])], m)[0].frame_idxs
    assert 1000 < f[1] < 9000, f


def test_trake_don_cuc_thi_rai_deu():
    """N sự kiện KHÔNG THỂ nằm trong vài phần trăm giây.

    Bản đầu điền `khung_trước + 1` cho ra 564,565,566 — dồn hết cơ hội vào một
    điểm. Dồn cục thì rải đều, cho mỗi sự kiện một cửa độc lập.
    """
    m = master_gia()
    ds = R.dung_trake([uv("L01_V000", [3000])] * 3, m)
    f = ds[0].frame_idxs
    assert f[-1] - f[0] > R.DON_NHAU, f"vẫn dồn cục: {f}"


def test_trake_video_co_su_kien_dung_TRUOC_video_bu():
    """Video có sự kiện phải xếp trước; video bù chỉ để lấp chỗ trống.

    Bù là có chủ ý: TRAKE chấm TỪNG PHẦN và không có điểm phạt, nên để trống
    75/100 dòng là vứt không 75 cơ hội. Nhưng thứ tự phải đúng — R@1 chiếm 1/5
    tổng điểm, không được để dòng bù chen lên đầu.
    """
    m = master_gia()
    ds = R.dung_trake([uv("L01_V000", [1000]), uv("L01_V000", [2000])], m)
    assert ds[0].video_id == "L01_V000"
    assert len(ds) > 1, "phải bù thêm dòng, không để trống"


def test_trake_bu_cho_du_so_dong():
    """Không để trống slot: dòng thứ 100 vẫn đáng 0,2 nếu trúng một vị trí."""
    m = master_gia(n_video=200, n_khung=10)
    ds = R.dung_trake([uv("L01_V000", [1000])] * 2, m, so_dong=100)
    assert len(ds) == 100
    assert all(len(x.frame_idxs) == 2 for x in ds)
    assert all(list(x.frame_idxs) == sorted(x.frame_idxs) for x in ds)


def test_trake_khong_vuot_100_dong():
    m = master_gia(n_video=3)
    nhieu = [Candidate(0, f"L01_V{i % 3:03d}", i * 100, 1.0) for i in range(300)]
    assert len(R.dung_trake([nhieu, nhieu], m, so_dong=100)) <= 100


def test_trake_qua_duoc_bo_soat_cua_nop_bai():
    """Chốt nối: thứ `dung_trake` sinh ra phải được `nop_bai.soat` chấp nhận."""
    from nop_bai import soat
    m = master_gia()
    ds = R.dung_trake([uv("L01_V000", [5000]), uv("L01_V000", [1000]),
                       uv("L01_V000", [3000])], m)
    assert not soat("query-4-trake", ds, so_su_kien=3)


# ---- dong_hang_dp (Bước 5) — sửa lỗi sorted() hoán đổi nhầm sự kiện -------

def test_dp_khong_hoan_doi_khung_giua_hai_su_kien():
    """Lỗi thật đã sửa: sự kiện 0 chỉ có ứng viên 5000, sự kiện 1 chỉ có ứng
    viên 1000 (NHỎ HƠN). `sorted([5000, 1000])` sẽ cho sự kiện 0 nhận nhầm
    giá trị 1000 (của sự kiện 1). DP không được làm vậy: không ép được chuỗi
    tăng dần hợp lệ thì để None, không hoán đổi bừa.
    """
    ra = R.dong_hang_dp([[(5000, 1.0)], [(1000, 1.0)]])
    assert ra[1] == 1000, ra
    assert ra[0] != 1000, f"hoán đổi nhầm: sự kiện 0 nhận khung của sự kiện 1: {ra}"


def test_dp_chon_ung_vien_hang_hai_khi_hang_mot_khong_tang_dan_duoc():
    """Sự kiện 0 có 2 ứng viên: 5000 (điểm cao) và 500 (điểm thấp). Chỉ 500
    mới tăng dần được với khung 1000 của sự kiện 1 — DP phải chọn 500, không
    phải rank-1 (5000), để giữ được chuỗi tăng dần."""
    ra = R.dong_hang_dp([[(5000, 0.9), (500, 0.5)], [(1000, 1.0)]])
    assert ra == [500, 1000], ra


def test_dp_giu_thu_tu_ngay_ca_khi_tang_dan_san():
    """Khi ứng viên đã tăng dần sẵn theo đúng thứ tự sự kiện, DP giữ nguyên
    liên kết — không có gì để hoán đổi."""
    ra = R.dong_hang_dp([[(1000, 1.0)], [(2000, 1.0)], [(3000, 1.0)]])
    assert ra == [1000, 2000, 3000]


def test_dp_su_kien_khong_co_ung_vien_ra_none():
    ra = R.dong_hang_dp([[(1000, 1.0)], [], [(3000, 1.0)]])
    assert ra == [1000, None, 3000]


def test_dp_tat_ca_rong_ra_toan_none():
    assert R.dong_hang_dp([[], []]) == [None, None]


# ---- lam_day_bang_trich_day (Bước 4) ---------------------------------------

class _KenhGia:
    """Giả `KenhAnh` — không nạp model thật, trả điểm số cố định để kiểm
    logic ghép ứng viên, không kiểm chất lượng encode (đã kiểm riêng ở
    test_dense.py bằng model thật). `vec @ q` phải ra đúng "diem" gắn sẵn
    trên từng KhungDay giả -> vector 1 chiều [diem], truy vấn 1 chiều [1.0]."""

    def encode_text(self, cau):
        import numpy as np
        return np.array([1.0], dtype=np.float32)

    def encode_image(self, anh):
        import numpy as np
        return np.array([[a.diem] for a in anh], dtype=np.float32)


class _KhungGia:
    def __init__(self, frame_idx, diem):
        self.frame_idx = frame_idx
        self.diem = diem
        self.anh = self   # __matmul__ ở _KenhGia đọc lại .diem qua chính object


def test_trich_day_them_ung_vien_moi_khong_trung_frame_cu(monkeypatch):
    """Khung trích dày mới phải được nối vào, không đè lên ứng viên cũ."""
    import trich_day as td

    def fake_trich_day(video_path, vid, center_frame, center_pts_time, fps,
                       radius_sec=2.0, stride=2, cache_dir="cache"):
        return [_KhungGia(1500, 0.5), _KhungGia(2500, 0.9)]

    monkeypatch.setattr(td, "trich_day", fake_trich_day)

    master = master_gia()   # L01_V000 có sẵn frame_idx=2000 (i=4, xem master_gia)
    ung_vien = [[(2000, 0.4)]]   # neo duy nhất của sự kiện 0
    R.lam_day_bang_trich_day("L01_V000", "gia.mp4", ["câu hỏi"], ung_vien,
                             _KenhGia(), master)
    frames = {f for f, _ in ung_vien[0]}
    assert frames == {2000, 1500, 2500}, ung_vien


def test_trich_day_bo_qua_su_kien_khong_co_neo(monkeypatch):
    """Sự kiện không có ứng viên nào trong video này thì không trích gì cả —
    không có neo để biết trích quanh đâu."""
    import trich_day as td
    goi = {"lan": 0}

    def fake_trich_day(*a, **k):
        goi["lan"] += 1
        return []

    monkeypatch.setattr(td, "trich_day", fake_trich_day)
    master = master_gia()
    ung_vien = [[], [(1000, 0.5)]]
    R.lam_day_bang_trich_day("L01_V000", "gia.mp4", ["c1", "c2"], ung_vien,
                             _KenhGia(), master)
    assert goi["lan"] == 1, "phải gọi trích dày đúng 1 lần (sự kiện có neo)"
