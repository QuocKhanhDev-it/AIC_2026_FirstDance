"""Test KHOI dau-cuoi cho `run.py`: thu muc de -> file CSV tren dia.

VI SAO CAN, DU DA CO 30 TEST CHO run.py
=======================================

Tu A51 toi luc phat hien (nhieu tuan), `run.py` **khong sinh noi mot file CSV
nao** cho bat ky goi nao co truy van > 40 tu: `hoi()` goi `hop_nhat()` ma
khong co import o tang module. 19/25 goi `De_Thi_Chinh_Thuc` va 18/24 goi
`de_thi_thu` dinh loi. Suot thoi gian do **328 test van xanh**.

Ly do bo test cu khong bat duoc: chung chi cham HAM THUAN (`tach_truy_van`,
`dong_hang_dp`, `_don_cuc`...) hoac goi `quet_anh` voi kenh gia. Khong test nao
di het duong `de/ -> quet -> dong_goi -> ghi_goi -> CSV`.

Va ky luat do cua repo cung khong bat duoc: no chi hoi "cau hinh nao TOT HON",
khong bao gio hoi "cau hinh nay CHAY NOI khong". Moi script do deu tu goi
`hop_nhat` cua rieng no.

Test nay dung mot `index/` TI HON (24 dong, ma tran 8 chieu) nen chay tren moi
may, khong can du lieu that. No khong do CHAT LUONG — no chi hoi mot cau:
**chay het duong co ra file khong.**
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

# Truy van DAI (> TRAN_TOKEN = 40 tu) -> bi tach menh de -> di vao dung nhanh
# tung lam run.py chet. Do la ca duy nhat quan trong o day.
CAU_DAI = ("Canh quay mot nhom hon nam nguoi xep thanh hang doc tren san co "
           "rong, ho cung thuc hien dong tac gio hai tay len cao roi ha xuong "
           "cham mui chan, phia sau la mot day nha mai ngoi mau do va vai cay "
           "co thu, troi nang, co bong nguoi in xuong mat san, may quay lia "
           "tu trai sang phai roi dung lai o giua khung hinh.")
CAU_NGAN = "mot con meo vang"


def _dung_index(d: Path, n=24, chieu=8):
    """Index ti hon nhung DU THAT de moi module doc duoc nhu binh thuong."""
    rng = np.random.default_rng(0)
    m = pd.DataFrame({
        "row_id": range(n),
        "video_id": [f"L01_V{i // 12 + 1:03d}" for i in range(n)],
        "kf_n": [i % 12 + 1 for i in range(n)],
        "kf_name": [f"{i % 12 + 1:03d}.jpg" for i in range(n)],
        "kf_path": [None] * n,
        "video_path": [None] * n,
        "pts_time": [float(i % 12) * 5.0 for i in range(n)],
        "frame_idx": [(i % 12) * 125 for i in range(n)],
        "fps": [25.0] * n,
        "title": ["ban tin"] * n,
        "description": ["mo ta"] * n,
        "keywords": ["tu khoa"] * n,
    })
    m.to_parquet(d / "master.parquet", index=False)

    mat = rng.standard_normal((n, chieu)).astype("float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    np.save(d / "clip_gopt.npy", mat)
    (d / "clip_gopt.json").write_text(
        json.dumps({"model": "gia", "pretrained": "gia", "chieu": chieu}),
        encoding="utf-8")

    # ⚠️ Chu HOA la bat buoc, khong phai trang tri: `dap_an.TEN` chi bat cum
    # bat dau bang chu hoa (A84). Van ban toan chu thuong -> bo dao ra RONG ->
    # moi dong nhan dap an mac dinh, va test rai bien the se do FIXTURE chu
    # khong do MA.
    pd.DataFrame({
        "row_id": range(n),
        "text": [f"Nguyen Van An Tran Thi Bich Le Hoang Nam nhom nguoi san co {i}"
                 for i in range(n)],
    }).to_parquet(d / "ocr_asr.parquet", index=False)
    return m


def _dung_cache(d: Path, cac_cau: list, chieu=8):
    """Cache vector truy van — PHAI co du MOI menh de, khong thi KeyError."""
    import run as R
    chuoi = []
    for c in cac_cau:
        chuoi.append(c)
        chuoi += R.tach_truy_van(c)
        for sk in R.tach_su_kien(c):
            chuoi.append(sk)
            chuoi += R.tach_truy_van(sk)
    chuoi = list(dict.fromkeys(x.strip() for x in chuoi if x.strip()))

    rng = np.random.default_rng(1)
    v = rng.standard_normal((len(chuoi), chieu)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.savez_compressed(
        d / "truy_van_gopt.npz",
        cau=np.array(chuoi, dtype=object).astype(str), vec=v,
        ghi_chu=json.dumps({"model": "gia", "pretrained": "gia",
                            "matrix": "clip_gopt.npy", "chieu": chieu}))
    return chuoi


@pytest.fixture
def bo_de(tmp_path):
    idx = tmp_path / "index"
    idx.mkdir()
    _dung_index(idx)

    de = tmp_path / "de"
    de.mkdir()
    (de / "query-1-kis.txt").write_text(CAU_DAI, encoding="utf-8")
    (de / "query-2-qa.txt").write_text(CAU_DAI, encoding="utf-8")
    (de / "query-3-kis.txt").write_text(CAU_NGAN, encoding="utf-8")
    trake = f"{CAU_DAI}\nE1: {CAU_NGAN} chay ra\nE2: no dung lai\nE3: no nam xuong"
    (de / "query-4-trake.txt").write_text(trake, encoding="utf-8")

    _dung_cache(idx, [CAU_DAI, CAU_NGAN, trake])
    return tmp_path, idx, de


def _chay(tmp_path, idx, de, *them):
    ra = tmp_path / "nop"
    p = subprocess.run(
        [sys.executable, str(GOC / "src" / "run.py"),
         "--de", str(de), "--ra", str(ra), "--index", str(idx),
         "--cache", str(idx / "truy_van_gopt.npz"), *them],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(GOC), timeout=300)
    return ra, p


def test_sinh_du_CSV_cho_moi_goi_ke_ca_cau_dai(bo_de):
    """Ca duy nhat tung hong: cau > 40 tu -> tach menh de -> NameError.

    `quet_anh` chay TRUOC vong lap goi, nen mot goi hong giet CA LUOT va khong
    file nao duoc ghi — khong phai mat mot goi.
    """
    tmp_path, idx, de = bo_de
    ra, p = _chay(tmp_path, idx, de)
    assert p.returncode == 0, f"run.py chet:\n{p.stdout[-3000:]}\n{p.stderr[-2000:]}"
    csv = sorted(x.name for x in ra.glob("*.csv"))
    assert len(csv) == 4, f"cho 4 goi, ra {len(csv)}: {csv}\n{p.stdout[-2000:]}"
    for f in ra.glob("*.csv"):
        assert f.stat().st_size > 0, f"{f.name} rong"


def test_TRAKE_moi_dong_TANG_DAN_NGAT_khong_trung_frame(bo_de):
    """A5.7: pts_time tang dan KHONG bao dam frame_idx tang dan.

    614 cap keyframe cung video co pts tang ma frame_idx BANG NHAU. Nop hai su
    kien o cung mot Frame ID la phi mot trong hai, va `nop_bai.soat` khong bat
    duoc vi no so voi `sorted()` — ma [0, 0, 519] thi da sorted.
    """
    import csv as _csv
    tmp_path, idx, de = bo_de
    ra, p = _chay(tmp_path, idx, de)
    assert p.returncode == 0, p.stdout[-2000:]
    f = ra / "query-4-trake.csv"
    assert f.exists()
    n = 0
    for row in _csv.reader(f.open(encoding="utf-8")):
        fx = [int(x) for x in row[1:] if x.strip().lstrip("-").isdigit()]
        if len(fx) < 2:
            continue
        n += 1
        assert fx == sorted(fx), f"khong tang dan: {fx}"
        assert len(set(fx)) == len(fx), f"TRUNG frame_idx: {fx}"
    assert n > 0, "khong doc duoc dong TRAKE nao"


def test_rai_bien_the_thuc_su_ra_nhieu_answer_khac_nhau(bo_de):
    """`tu_ung_vien` tung bo trung theo (video, frame) nen vut sach bien the.

    A83 do rai bien the gap 2,3 lan diem Q&A — nhung do o script KHONG di qua
    `tu_ung_vien`, nen cho hong khong lo ra.
    """
    import csv as _csv
    tmp_path, idx, de = bo_de
    ra, p = _chay(tmp_path, idx, de, "--rai-bien-the", "2")
    assert p.returncode == 0, p.stdout[-2000:]
    dong = list(_csv.reader((ra / "query-2-qa.csv").open(encoding="utf-8")))
    khoa2 = {(r[0], r[1]) for r in dong if len(r) >= 3}
    khoa3 = {(r[0], r[1], r[2]) for r in dong if len(r) >= 3}
    assert len(khoa3) > len(khoa2), (
        "khong dong nao dung chung khung ma khac `answer` -> bien the bi vut")
