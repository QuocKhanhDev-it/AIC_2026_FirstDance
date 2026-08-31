"""
test_van_ban_dense.py — Chốt cho kênh 6 (OCR/ASR nhúng vector).

Ba thứ đáng chốt, tất cả đều là loại hỏng KHÔNG ném lỗi:
  * gộp nhiều đoạn của một keyframe bằng MAX, không phải trung bình
  * lệch cặp file (vector văn bản và cache truy vấn khác model) phải DỪNG
  * bể ứng viên khoá được
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from van_ban_dense import KenhVanBanDense  # noqa: E402

CHIEU = 8


def _chuan(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


@pytest.fixture
def kho(tmp_path):
    """3 keyframe; row 1 có HAI đoạn — một khớp truy vấn, một không."""
    m = pd.DataFrame({
        "row_id": [0, 1, 2],
        "video_id": ["L21_V001"] * 3,
        "frame_idx": [10, 20, 30],
        "pts_time": [1.0, 2.0, 3.0],
        "fps": [25.0] * 3,
        "kf_n": [1, 2, 3],
    })
    m.to_parquet(tmp_path / "master.parquet", index=False)

    q = _chuan(np.eye(CHIEU)[0])                    # truy vấn = trục 0
    doan = _chuan(np.array([
        np.eye(CHIEU)[1],       # row 0 — không liên quan
        np.eye(CHIEU)[0],       # row 1, đoạn A — khớp hoàn toàn
        np.eye(CHIEU)[2],       # row 1, đoạn B — không liên quan
        np.eye(CHIEU)[3],       # row 2 — không liên quan
    ]))
    np.savez(tmp_path / "van_ban.npz", vec=doan.astype(np.float16),
             row_id=np.array([0, 1, 1, 2], dtype=np.int64),
             ghi_chu=json.dumps({"model": "gopt", "chieu": CHIEU}))
    np.savez(tmp_path / "cache.npz", cau=np.array(["tìm cái này"]),
             vec=q[None, :].astype(np.float32),
             ghi_chu=json.dumps({"model": "gopt", "chieu": CHIEU}))
    return tmp_path


def test_gop_bang_max_khong_phai_trung_binh(kho):
    """row 1 có một đoạn khớp hoàn toàn và một đoạn trực giao.

    Gộp bằng MAX -> row 1 hạng 1. Gộp bằng trung bình -> điểm nó tụt còn 0,5
    và bị mọi row khác vượt. Đây là khác biệt quyết định của kênh này.
    """
    k = KenhVanBanDense(str(kho), kho / "van_ban.npz", kho / "cache.npz")
    kq = k.tim("tìm cái này", k=3)
    assert kq[0].row_id == 1
    assert kq[0].score == pytest.approx(1.0, abs=0.01)


def test_moi_row_chi_ra_MOT_ung_vien(kho):
    """row 1 có hai đoạn nhưng chỉ được xuất hiện một lần — nếu không, RRF
    đếm nó hai lần và tự cộng hưởng với chính mình."""
    k = KenhVanBanDense(str(kho), kho / "van_ban.npz", kho / "cache.npz")
    r = [c.row_id for c in k.tim("tìm cái này", k=10)]
    assert len(r) == len(set(r)) == 3


def test_dien_du_video_id_va_frame_idx(kho):
    """`frame_idx` là giá trị nộp cho BTC — phải lấy từ bảng cái, không để
    trống rồi tính lại ở tầng trên."""
    k = KenhVanBanDense(str(kho), kho / "van_ban.npz", kho / "cache.npz")
    c = k.tim("tìm cái này", k=1)[0]
    assert c.video_id == "L21_V001" and c.frame_idx == 20
    assert c.source == "van_ban_dense"


def test_khoa_be_ung_vien(kho):
    k = KenhVanBanDense(str(kho), kho / "van_ban.npz", kho / "cache.npz")
    be = np.array([True, False, True])          # loại row 1
    r = [c.row_id for c in k.tim("tìm cái này", k=10, be=be)]
    assert 1 not in r and set(r) == {0, 2}


def test_lech_so_chieu_thi_dung(kho, tmp_path):
    """Cache 1152 chiều + văn bản 8 chiều = sai cặp file. Phải DỪNG, không
    được lặng lẽ chạy ra kết quả vô nghĩa."""
    xau = tmp_path / "cache_xau.npz"
    np.savez(xau, cau=np.array(["x"]),
             vec=np.zeros((1, 1152), dtype=np.float32),
             ghi_chu=json.dumps({"model": "SO400M", "chieu": 1152}))
    with pytest.raises(SystemExit, match="Sai cặp file"):
        KenhVanBanDense(str(kho), kho / "van_ban.npz", xau)


def test_truy_van_ngoai_cache_thi_nem_loi(kho):
    """Không đoán bừa: trả vector 0 sẽ cho 100 ứng viên ngẫu nhiên trông hợp lệ."""
    k = KenhVanBanDense(str(kho), kho / "van_ban.npz", kho / "cache.npz")
    with pytest.raises(KeyError):
        k.tim("câu chưa từng mã hoá", k=5)


def test_vec_va_row_id_lech_thi_dung(kho, tmp_path):
    xau = tmp_path / "hong.npz"
    np.savez(xau, vec=np.zeros((4, CHIEU), dtype=np.float16),
             row_id=np.array([0, 1], dtype=np.int64),
             ghi_chu=json.dumps({"model": "gopt"}))
    with pytest.raises(SystemExit, match="file hỏng"):
        KenhVanBanDense(str(kho), xau, kho / "cache.npz")
