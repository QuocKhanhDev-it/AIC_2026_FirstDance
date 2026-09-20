"""Kiem `web/server.py` — giao dien phai nhin CUNG bo ung vien voi bai nop.

Day la cho lech nguy hiem nhat cua ca he thong: nguoi soat nhin mot be ung
vien roi gui di mot be khac. Ba test duoi day chot ba cho da tung lech that.

Khong can `index/` — dung kenh gia.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

import server as S                                    # noqa: E402
from rrf import hop_nhat                              # noqa: E402
from schema import Candidate                          # noqa: E402


class KenhGia:
    """Tra ve ung vien khac nhau cho tung menh de, de bat cach hop nhat."""

    def __init__(self, theo_menh_de):
        self.theo_menh_de = theo_menh_de
        self.da_goi = []

    def tim(self, cau, k=100, **kw):
        self.da_goi.append(cau)
        khoa = cau[0] if isinstance(cau, list) else cau
        return self.theo_menh_de.get(khoa, [])


def uv(*bo):
    return [Candidate(r, f"V{r}", r * 10, s, "gia") for r, s in bo]


def test_trong_so_khop_run_py():
    """A52 do kenh 3 o trong so 0,5. De 1:1 la ve be ung vien KHAC bai nop."""
    import argparse
    import run as R
    ap = argparse.ArgumentParser()
    ap.add_argument("--trong-so-phu", type=float, default=0.5)
    assert S.TRONG_SO["ocr"] == ap.parse_args([]).trong_so_phu
    assert S.TRONG_SO["anh"] == 1.0
    from rrf import K_MAC_DINH
    assert S.RRF_K == K_MAC_DINH


def test_hop_nhat_menh_de_bang_RRF_HANG_khong_phai_max_cosine():
    """A51: max cosine THUA RRF hang -0,0721, ✅ on dinh.

    `dense.KenhAnhCache.tim(danh_sach)` lay np.max qua cac menh de — dung cach
    da bi bac. `Kho._hoi` phai goi TUNG menh de roi hop nhat theo hang.
    """
    kho = S.Kho.__new__(S.Kho)               # khong chay __init__ (can index/)
    k = KenhGia({"A": uv((1, 0.9), (2, 0.8)), "B": uv((3, 0.7), (1, 0.6))})

    import run as R
    that = R.tach_truy_van
    R.tach_truy_van = lambda c, **kw: ["A", "B"]
    try:
        ra = kho._hoi(k, "khong quan trong", 10)
    finally:
        R.tach_truy_van = that

    assert k.da_goi == ["A", "B"], "phai goi TUNG menh de, khong goi ca danh sach"
    mong = hop_nhat([uv((1, 0.9), (2, 0.8)), uv((3, 0.7), (1, 0.6))])[:10]
    assert [c.row_id for c in ra] == [c.row_id for c in mong]


def test_mot_menh_de_thi_goi_thang_khong_qua_RRF():
    """Hop nhat MOT danh sach chi lam mat diem goc — run.py cung lam vay."""
    kho = S.Kho.__new__(S.Kho)
    k = KenhGia({"A": uv((5, 0.9), (6, 0.8))})
    import run as R
    that = R.tach_truy_van
    R.tach_truy_van = lambda c, **kw: ["A"]
    try:
        ra = kho._hoi(k, "x", 10)
    finally:
        R.tach_truy_van = that
    assert [c.row_id for c in ra] == [5, 6]
    assert ra[0].score == 0.9, "diem goc phai giu nguyen, khong doi thanh RRF"
