"""
test_anh.py — Chốt cho luật lùi về bản thu nhỏ.

Luật này quyết định người soi bài nhìn thấy gì. Sai theo hướng "im lặng trả
bản thu nhỏ mà không nói" là tệ nhất: bản 256px đủ để nhận ra cảnh nhưng KHÔNG
đọc được chữ nhỏ, mà phần lớn câu Q&A đề thật là câu đọc chữ.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import anh as ANH  # noqa: E402


def _anh(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff\xd8\xff\xe0 gia lam anh")
    return p


def test_uu_tien_anh_goc(tmp_path):
    goc = _anh(tmp_path / "goc" / "007.jpg")
    _anh(ANH.duong_ban_nho("L21_V001", 7, tmp_path / "nho"))
    p, nho = ANH.tim(str(goc), "L21_V001", 7, tmp_path / "nho")
    assert p == goc and nho is False


def test_lui_ve_ban_nho_khi_khong_co_goc(tmp_path):
    n = _anh(ANH.duong_ban_nho("L26_V001", 1, tmp_path / "nho"))
    p, nho = ANH.tim(None, "L26_V001", 1, tmp_path / "nho")
    assert p == n and nho is True


def test_kf_path_tro_vao_file_khong_ton_tai_van_lui_duoc(tmp_path):
    """`kf_path` là đường dẫn của MÁY DỰNG INDEX (A5.5) — trên máy khác nó trỏ
    vào chỗ không có gì. Phải lùi, không được coi như đã có ảnh."""
    n = _anh(ANH.duong_ban_nho("L26_V002", 3, tmp_path / "nho"))
    p, nho = ANH.tim(r"D:\Project\khong_co_o_day\003.jpg", "L26_V002", 3,
                     tmp_path / "nho")
    assert p == n and nho is True


def test_khong_co_gi_thi_tra_none(tmp_path):
    p, nho = ANH.tim(None, "L26_V003", 9, tmp_path / "nho")
    assert p is None and nho is False


def test_ten_file_dem_ba_chu_so(tmp_path):
    """Cây ảnh gốc của BTC đánh số `001.jpg`. Bản thu nhỏ phải trùng khuôn đó,
    nếu không `12_va_duong_dan.py` quét được thư mục mà khớp trượt từng dòng."""
    assert ANH.duong_ban_nho("L26_V001", 7, tmp_path).name == "007.jpg"
    assert ANH.duong_ban_nho("L26_V001", 123, tmp_path).name == "123.jpg"


def test_thong_ke_khong_co_thu_muc_nho(tmp_path):
    m = pd.DataFrame({"video_id": ["L21_V001", "L26_V001"],
                      "kf_path": ["/co/that.jpg", None]})
    tk = ANH.thong_ke(m, tmp_path / "chua_co")
    assert tk["anh_goc"] == 1 and tk["ban_nho"] == 0
    assert tk["tong_soi_duoc"] == 1 and tk["tong_dong"] == 2


def test_thong_ke_dem_them_ban_nho(tmp_path):
    m = pd.DataFrame({"video_id": ["L21_V001", "L26_V001", "L26_V001"],
                      "kf_path": ["/co/that.jpg", None, None]})
    nho = tmp_path / "nho"
    _anh(ANH.duong_ban_nho("L26_V001", 1, nho))
    _anh(ANH.duong_ban_nho("L26_V001", 2, nho))
    tk = ANH.thong_ke(m, nho)
    assert tk["anh_goc"] == 1 and tk["ban_nho"] == 2
    assert tk["tong_soi_duoc"] == 3


def test_thong_ke_khong_dem_qua_so_dong_that(tmp_path):
    """Thư mục thu nhỏ có thể thừa file (video đã xoá khỏi bảng cái). Không
    được đếm quá số dòng thật, nếu không `tong_soi_duoc` vượt `tong_dong`."""
    m = pd.DataFrame({"video_id": ["L26_V001"], "kf_path": [None]})
    nho = tmp_path / "nho"
    for i in range(1, 6):
        _anh(ANH.duong_ban_nho("L26_V001", i, nho))
    tk = ANH.thong_ke(m, nho)
    assert tk["ban_nho"] == 1
    assert tk["tong_soi_duoc"] <= tk["tong_dong"]
