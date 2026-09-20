"""
test_cua_so.py — Chốt cho cách chấm điểm theo cửa sổ (A38).

Hai bất biến quan trọng nhất, vì cả hai đều hỏng ÂM THẦM:

* cửa sổ **không được tràn sang video khác** — bảng cái sắp liền nhau nên hàng
  xóm ở biên video là một video hoàn toàn khác;
* điểm cửa sổ phải **thưởng cho vùng phủ nhiều mệnh đề**, chứ không phải cho
  một khung khớp mạnh một mệnh đề — đó là toàn bộ lý do nó tồn tại.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from cua_so import bien_video, diem_cua_so, diem_khung_roi   # noqa: E402


def test_bien_video_danh_so_theo_nhom():
    m = pd.DataFrame({"video_id": ["A", "A", "B", "B", "B"]})
    assert list(bien_video(m)) == [0, 0, 1, 1, 1]


def test_cua_so_cong_duoc_hai_menh_de_o_hai_khung_khac_nhau():
    """Ca query-p1-4-kis: mệnh đề 1 ở khung 0, mệnh đề 2 ở khung 2.

    Không khung nào khớp cả hai, nhưng khung 1 nằm giữa nên cửa sổ của nó phủ
    được cả hai — và phải thắng.
    """
    sim = np.array([[1.0, 0.0, 0.0],      # mệnh đề 1 chỉ khớp khung 0
                    [0.0, 0.0, 1.0]])     # mệnh đề 2 chỉ khớp khung 2
    nhom = np.array([0, 0, 0])
    d = diem_cua_so(sim, nhom, ban_kinh=1)
    assert d[1] == max(d)                 # khung giữa phủ cả hai -> thắng
    assert d[1] == 2.0


def test_khung_roi_khong_lam_duoc_dieu_do():
    """Mốc nền: lấy max qua mệnh đề thì ba khung hoà nhau, không phân định được."""
    sim = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    d = diem_khung_roi(sim)
    assert d[1] == 0.0                    # khung giữa không khớp mệnh đề nào
    assert list(d) == [1.0, 0.0, 1.0]


def test_cua_so_KHONG_tran_sang_video_khac():
    """Bất biến quan trọng nhất — bảng cái sắp liền nhau nên rất dễ tràn."""
    sim = np.array([[0.0, 9.0]])          # khung 1 điểm cao
    nhom = np.array([0, 1])               # nhưng thuộc video KHÁC
    d = diem_cua_so(sim, nhom, ban_kinh=3)
    assert d[0] == 0.0                    # không được mượn điểm của video khác


def test_ban_kinh_0_thi_bang_tong_qua_menh_de():
    sim = np.array([[1.0, 2.0], [3.0, 4.0]])
    nhom = np.array([0, 0])
    assert list(diem_cua_so(sim, nhom, ban_kinh=0)) == [4.0, 6.0]


def test_nhan_mot_menh_de_duy_nhat():
    d = diem_cua_so(np.array([0.5, 0.1, 0.9]), np.array([0, 0, 0]), ban_kinh=1)
    assert d[1] == 0.9                    # khung giữa mượn được từ hàng xóm


def test_lech_do_dai_thi_bao_loi_ngay():
    import pytest
    with pytest.raises(ValueError):
        diem_cua_so(np.array([[1.0, 2.0]]), np.array([0, 0, 0]))
