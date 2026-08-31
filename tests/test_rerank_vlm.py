"""
test_rerank_vlm.py — Chốt cho cách hợp nhất điểm VLM (`scripts/64_`).

Bốn cách xếp lại trong `64_` quyết định phép đo Nấc 3 có nghĩa hay không, và
cả bốn đều hỏng theo kiểu KHÔNG ném lỗi:

  * làm rơi hoặc nhân đôi ứng viên  -> điểm đổi vì lý do chẳng liên quan
  * xếp lại luôn cả phần ĐUÔI VLM không chấm -> so hai thang điểm khác nhau,
    đúng lỗi RRF sinh ra để tránh
  * "chỉ đẩy lên" mà vẫn đẩy xuống được -> mất câu kênh 1 vốn đã làm đúng
"""

import importlib.util
import sys
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from schema import Candidate  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "do_rerank_vlm", GOC / "scripts" / "64_do_rerank_vlm.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

DAU = 5


def _ds(n=12):
    """n ứng viên, điểm giảm dần — giống đầu ra `hop_nhat`."""
    return [Candidate(row_id=i, video_id=f"L21_V{i:03d}", frame_idx=i * 10,
                      score=1.0 / (i + 1), source="rrf",
                      meta={"pts_time": float(i), "fps": 25.0, "kf_n": i})
            for i in range(n)]


def _vlm_dao_nguoc():
    """VLM thích ĐÚNG kẻ kênh 1 xếp bét trong phần đầu — trường hợp làm lộ
    mọi lỗi trộn vùng."""
    return {i: float(i) for i in range(DAU)}


CACH = [
    ("thay_han", lambda ds, v: M.thay_han(ds, v, DAU)),
    ("rrf_hang", lambda ds, v: M.rrf_hang(ds, v, DAU, 1.0)),
    ("nhan", lambda ds, v: M.nhan(ds, v, DAU, 1.0)),
    ("chi_day_len", lambda ds, v: M.nhan(ds, v, DAU, 1.0, chi_day_len=True)),
]


@pytest.mark.parametrize("ten,f", CACH)
def test_khong_roi_khong_nhan_doi(ten, f):
    ds = _ds()
    ra = f(ds, _vlm_dao_nguoc())
    assert [c.row_id for c in sorted(ra, key=lambda c: c.row_id)] == \
           [c.row_id for c in sorted(ds, key=lambda c: c.row_id)]


@pytest.mark.parametrize("ten,f", CACH)
def test_duoi_giu_nguyen_thu_tu_va_nam_sau(ten, f):
    """VLM chỉ chấm `DAU` ứng viên đầu. Phần đuôi không có điểm VLM nên không
    được xếp lẫn vào — và phải giữ đúng thứ tự cũ."""
    ds = _ds()
    ra = f(ds, _vlm_dao_nguoc())
    assert [c.row_id for c in ra[DAU:]] == [c.row_id for c in ds[DAU:]]
    assert {c.row_id for c in ra[:DAU]} == {c.row_id for c in ds[:DAU]}


def test_thay_han_theo_dung_diem_vlm():
    ds = _ds()
    ra = M.thay_han(ds, _vlm_dao_nguoc(), DAU)
    assert [c.row_id for c in ra[:DAU]] == [4, 3, 2, 1, 0]


def test_chi_day_len_khong_ha_ai_xuong():
    """Ứng viên bị VLM chấm THẤP phải giữ nguyên điểm, không bị trừ."""
    ds = _ds()
    vlm = {0: -5.0, 1: -5.0, 2: -5.0, 3: -5.0, 4: 10.0}   # chỉ #4 được thích
    ra = {c.row_id: c.score for c in M.nhan(ds, vlm, DAU, 1.0, chi_day_len=True)}
    goc = {c.row_id: c.score for c in ds}
    for rid in range(DAU):
        assert ra[rid] >= goc[rid] - 1e-9, f"row {rid} bị hạ điểm"
    assert ra[4] > goc[4]


def test_chuan_hoa_theo_TUNG_cau():
    """Chuẩn hoá toàn cục là sai: mỗi câu một độ khó, thang tuyệt đối không so
    được. Điểm cao nhất trong câu phải thành 1,0 dù giá trị thô là bao nhiêu."""
    assert M._chuan({1: -8.0, 2: -4.0, 3: 0.0}) == {1: 0.0, 2: 0.5, 3: 1.0}
    assert M._chuan({1: 3.0, 2: 3.0}) == {1: 0.5, 2: 0.5}   # bằng nhau -> 0,5
    assert M._chuan({}) == {}


def test_thieu_diem_vlm_thi_khong_no():
    """Ảnh thiếu trên máy chấm -> ứng viên không có điểm. Phải chạy tiếp chứ
    không ném KeyError giữa một phép đo 52 câu."""
    ds = _ds()
    for _, f in CACH:
        ra = f(ds, {0: 1.0})            # chỉ 1/5 ứng viên đầu có điểm
        assert len(ra) == len(ds)
