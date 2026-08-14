"""
test_trich_day.py — Chốt chặn cho TV2 (đường găng).

Ba việc phải chứng minh, không chỉ "chạy không lỗi":

1. `frame_idx` trả về đúng CHÍNH XÁC dãy ta yêu cầu (center ± k*stride) — đây
   là số sẽ NỘP CHO BTC nếu khung này được chọn, sai một đơn vị là mất điểm.
2. Ảnh trích ở đúng `center_frame` (trùng một keyframe có sẵn) khớp pixel với
   ảnh keyframe gốc của BTC — cùng phép đo tương quan `02_verify.py` đã dùng
   để chứng minh bảng cái. Không có phép này thì không biết `center_pts_time`
   + `stride/fps` có neo đúng chỗ hay chỉ đúng may rủi.
3. Cache hoạt động: gọi lại lần hai KHÔNG spawn thêm ffmpeg nào.

Chạy:
    pytest tests/test_trich_day.py -v

Cần ffmpeg trong PATH và ít nhất một video .mp4 tồn tại thật trên máy (cột
video_path của master.parquet trỏ tới file có sẵn) — nếu không có gì thì cả
file bị skip, giống cách test_dense.py skip khi thiếu clip.npy.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

pytestmark = [
    pytest.mark.skipif(not (GOC / "index" / "master.parquet").exists(),
                        reason="chưa có index/master.parquet trên máy này"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None,
                        reason="cần ffmpeg trong PATH"),
]


def _hang(anh: Image.Image) -> np.ndarray:
    """Chữ ký ảnh xám 64x64 chuẩn hóa, cùng cách 02_verify.py dùng để so khớp."""
    a = np.asarray(anh.convert("L").resize((64, 64)), dtype=float)
    return (a - a.mean()) / (a.std() + 1e-6)


@pytest.fixture(scope="module")
def mot_video():
    """Một dòng master.parquet có CẢ video_path lẫn kf_path tồn tại thật trên
    máy này, neo giữa video (tránh biên đầu/cuối)."""
    m = pd.read_parquet(GOC / "index" / "master.parquet")
    m = m[m.video_path.notna() & m.kf_path.notna()]
    for vid in m.video_id.unique():
        sub = m[m.video_id == vid].sort_values("kf_n").reset_index(drop=True)
        r = sub.iloc[len(sub) // 2]
        if Path(r.video_path).exists() and Path(r.kf_path).exists():
            return r
    pytest.skip("Không có video_id nào có cả .mp4 lẫn ảnh keyframe tồn tại "
               "thật trên máy này.")


@pytest.fixture(scope="module")
def kenh():
    from trich_day import trich_day
    return trich_day


def test_frame_idx_dung_day_yeu_cau(kenh, mot_video, tmp_path):
    """Dãy frame_idx trả về phải KHỚP TUYỆT ĐỐI center ± k*stride."""
    r = mot_video
    stride = 5
    khung = kenh(r.video_path, r.video_id, int(r.frame_idx), float(r.pts_time),
                float(r.fps), radius_sec=1.0, stride=stride, cache_dir=tmp_path)
    assert khung, "không trích được khung nào — kiểm tra ffmpeg/đường dẫn video"

    idx = [k.frame_idx for k in khung]
    assert idx == sorted(idx), "phải sắp theo thời gian tăng dần"
    assert int(r.frame_idx) in idx, "phải kèm chính center_frame"
    for a, b in zip(idx, idx[1:]):
        assert b - a == stride, f"khoảng cách {b - a} != stride {stride}"


def test_khung_giua_khop_anh_keyframe_goc(kenh, mot_video, tmp_path):
    """Ảnh trích TẠI ĐÚNG center_frame phải khớp pixel với ảnh keyframe gốc
    của BTC — chứng minh việc neo bằng center_pts_time không bị lệch.

    Ngưỡng lấy từ 02_verify.py (MATCH = 0.95); dùng stride nhỏ + radius nhỏ để
    center_frame chắc chắn có trong danh sách trả về.
    """
    r = mot_video
    khung = kenh(r.video_path, r.video_id, int(r.frame_idx), float(r.pts_time),
                float(r.fps), radius_sec=0.2, stride=1, cache_dir=tmp_path)
    giua = next(k for k in khung if k.frame_idx == int(r.frame_idx))

    goc = Image.open(r.kf_path)
    corr = float((_hang(giua.anh) * _hang(goc)).mean())
    assert corr >= 0.95, (
        f"khung ở center_frame lệch khỏi keyframe gốc (corr={corr:.3f}) — "
        f"cách suy pts_time có thể sai")


def test_bo_qua_frame_idx_am(kenh, mot_video, tmp_path):
    """Center gần đầu video: không được trả frame_idx < 0."""
    r = mot_video
    khung = kenh(r.video_path, r.video_id, 2, float(r.pts_time) - 2 / float(r.fps),
                float(r.fps), radius_sec=1.0, stride=1, cache_dir=tmp_path)
    assert all(k.frame_idx >= 0 for k in khung)


def test_cache_khong_goi_lai_ffmpeg(kenh, mot_video, tmp_path, monkeypatch):
    """Gọi lần hai với cùng tham số phải đọc từ cache, KHÔNG spawn ffmpeg."""
    r = mot_video
    kwargs = dict(radius_sec=0.2, stride=1, cache_dir=tmp_path)
    lan_1 = kenh(r.video_path, r.video_id, int(r.frame_idx), float(r.pts_time),
                float(r.fps), **kwargs)
    assert lan_1, "cần trích được ít nhất một khung ở lần đầu để test có ý nghĩa"

    goi = {"so_lan": 0}
    that = subprocess.run

    def gia(*a, **kw):
        goi["so_lan"] += 1
        return that(*a, **kw)

    monkeypatch.setattr(subprocess, "run", gia)
    lan_2 = kenh(r.video_path, r.video_id, int(r.frame_idx), float(r.pts_time),
                float(r.fps), **kwargs)

    assert goi["so_lan"] == 0, "lần gọi thứ hai KHÔNG được spawn ffmpeg (phải trúng cache)"
    assert [k.frame_idx for k in lan_2] == [k.frame_idx for k in lan_1]


def test_tra_ve_khungday_khong_phai_candidate(kenh, mot_video, tmp_path):
    """Khung dày không có row_id — KHÔNG được lẫn với Candidate.

    Xem docstring đầu trich_day.py: row_id giả sẽ làm rrf.hop_nhat/
    dedup.gom_ban_sao tra nhầm vector CLIP của một keyframe khác.
    """
    r = mot_video
    khung = kenh(r.video_path, r.video_id, int(r.frame_idx), float(r.pts_time),
                float(r.fps), radius_sec=0.2, stride=1, cache_dir=tmp_path)
    assert khung and not hasattr(khung[0], "row_id")


def test_fps_khong_hop_le_nem_loi(kenh, mot_video, tmp_path):
    r = mot_video
    with pytest.raises(ValueError):
        kenh(r.video_path, r.video_id, int(r.frame_idx), float(r.pts_time),
            0.0, cache_dir=tmp_path)


def test_stride_duoi_1_nem_loi(kenh, mot_video, tmp_path):
    r = mot_video
    with pytest.raises(ValueError):
        kenh(r.video_path, r.video_id, int(r.frame_idx), float(r.pts_time),
            float(r.fps), stride=0, cache_dir=tmp_path)
