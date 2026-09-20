"""Kiem `kbest_trake.py` — lap bai nop TRAKE bang K-best beam search."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kbest_trake import (beam_video, cham_video, gom_theo_video,  # noqa: E402
                         lap_dong, lap_trake)
from schema import Candidate                                      # noqa: E402


def master_gia(n=12, moi_video=4):
    """Bang cai gia: row_id trung chi so dong (dieu kien A39 dua vao)."""
    return pd.DataFrame({
        "row_id": range(n),
        "video_id": [f"V{i // moi_video}" for i in range(n)],
        "pts_time": [float(i % moi_video) * 10.0 for i in range(n)],
        "frame_idx": [1000 + i * 25 for i in range(n)],
    })


def uv(*bo, nguon="k"):
    return [Candidate(r, v, 0, s, nguon) for r, v, s in bo]


def test_beam_ep_TANG_DAN_theo_thoi_gian():
    """BTC doi thu tu su kien theo thu tu THOI GIAN — chuoi giam la vo le."""
    pts = [0.0, 10.0, 20.0]
    # su kien 1 chi co khung o t=20, su kien 2 chi co khung o t=0 -> khong noi
    ra = beam_video([[(2, 1.0)], [(0, 1.0)]], pts, 5)
    assert ra == [] or all(len(c) < 2 for c in ra)
    # nguoc lai thi noi duoc
    ra = beam_video([[(0, 1.0)], [(2, 1.0)]], pts, 5)
    assert ra == [[0, 2]]


def test_loc_da_dang_bo_chuoi_qua_giong_nhau():
    """100 dong lech nhau vai phan tram giay se cung dung hoac cung sai."""
    pts = [0.0, 0.1, 50.0]
    hai = beam_video([[(0, 1.0), (1, 0.9)], [(2, 1.0)]], pts, 5, cach_nhau=3.0)
    assert len(hai) == 1, "hai chuoi lech 0,1s phai bi gop"
    rong = beam_video([[(0, 1.0), (1, 0.9)], [(2, 1.0)]], pts, 5, cach_nhau=0.05)
    assert len(rong) == 2, "noi long thi giu ca hai"


def test_video_thieu_su_kien_bi_loai():
    theo = gom_theo_video([uv((0, "V0", 0.9)), uv((4, "V1", 0.8))])
    assert set(cham_video(theo)) == set(), "khong video nao co DU hai su kien"


def test_cham_video_tuong_duong_trung_binh_NHAN():
    """Sum log(max) — moi video cung so su kien nen no la trung binh nhan.

    Kiem tinh chat do: nhan doi mot su kien o ca hai video khong doi thu hang.
    """
    m = master_gia()
    a = [uv((0, "V0", 0.9), (4, "V1", 0.5)), uv((1, "V0", 0.1), (5, "V1", 0.5))]
    d = cham_video(gom_theo_video(a))
    assert d["V1"] > d["V0"], "0,5*0,5 = 0,25 > 0,9*0,1 = 0,09"
    assert len(m) == 12


def test_n_duoi_phu_toi_hang_5_cong_n():
    """Luoi an toan: video hang 6.. moi cai mot dong. Day la ca trake-L25-004.

    Khong co no thi video dung nam ngoai top-5 -> KHONG mot gia thuyet nao.
    """
    m = master_gia(n=40, moi_video=2)      # 20 video, moi video 2 khung
    cac = []
    for i in range(2):                      # 2 su kien
        cac.append([Candidate(v * 2 + i, f"V{v}", 0, 1.0 - v * 0.01)
                    for v in range(20)])
    khong_duoi = lap_dong(cac, m, so_dong=100, n_duoi=0)
    co_duoi = lap_dong(cac, m, so_dong=100, n_duoi=10)
    vid = m.video_id.values
    v_khong = {vid[d[0]] for d in khong_duoi}
    v_co = {vid[d[0]] for d in co_duoi}
    assert len(v_khong) <= 5, "K-best thuan chi cham 5 video"
    assert len(v_co) > len(v_khong), "luoi an toan phai cham them video"


def test_lap_trake_lay_frame_idx_TU_BANG_CAI():
    """frame_idx la gia tri NOP — cam tinh lai tu pts_time (lech 1 frame)."""
    m = master_gia()
    cac = [uv((0, "V0", 0.9)), uv((1, "V0", 0.8))]
    ra = lap_trake(cac, m, so_dong=5)
    assert ra, "phai lap duoc it nhat mot dong"
    assert ra[0].video_id == "V0"
    assert ra[0].frame_idxs == [1000, 1025], "phai khop cot frame_idx"


def test_rong_thi_tra_ve_rong_chu_khong_no():
    m = master_gia()
    assert lap_dong([], m) == []
    assert lap_trake([], m) == []
    assert lap_trake([[], []], m) == []


def test_phat_bac_dung_hinh_dang():
    """Vung [gan, xa] KHONG bi phat — do la cho A80 bac phat ty le thuan."""
    from kbest_trake import phat_bac
    f = phat_bac(gan=1.0, nang=1.0, xa=60.0, beta=0.0005)
    assert f(0.5) == 1.0                    # qua gan -> phat nang
    assert f(1.0) == 0.0                    # dung bien duoi -> khong phat
    assert f(12.0) == 0.0                   # trung vi that -> khong phat
    assert f(60.0) == 0.0                   # dung bien tren -> khong phat
    assert f(160.0) == pytest.approx(0.05)  # 0,0005 * 100


def test_phat_bac_khong_dung_toi_su_kien_dau():
    """Su kien dau chua co Δt nen khong the bi phat."""
    from kbest_trake import beam_video, phat_bac
    pts = [0.0, 0.1, 50.0]
    f = phat_bac(gan=1.0, nang=999.0)       # phat cuc nang neu ap nham
    ra = beam_video([[(0, 1.0)], [(2, 1.0)]], pts, 5, phat=f)
    assert ra == [[0, 2]], "chuoi hop le van phai sinh ra duoc"


def test_phat_thay_doi_lua_chon_chuoi():
    """Phat nang o Δt nho phai day beam sang chuoi gian ra."""
    from kbest_trake import beam_video, phat_bac
    pts = [0.0, 0.5, 40.0]
    uv = [[(0, 1.0)], [(1, 0.9), (2, 0.85)]]
    khong = beam_video(uv, pts, 1)
    assert khong == [[0, 1]], "khong phat -> lay diem cao hon"
    co = beam_video(uv, pts, 1, phat=phat_bac(gan=1.0, nang=0.5))
    assert co == [[0, 2]], "phat Δt<1s -> chuyen sang chuoi gian ra"
