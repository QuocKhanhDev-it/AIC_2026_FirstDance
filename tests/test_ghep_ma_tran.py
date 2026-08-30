"""
test_ghep_ma_tran.py — Chốt cho việc ghép ma trận từ nhiều người encode.

Chỉ kiểm `kiem_cung_model`. Đó là chốt duy nhất đứng giữa "chia việc encode
cho 6 người" và "ghép nhầm hai không gian vector mà không có gì báo": shape và
dtype đều khớp trong ca đó, nên mọi phép kiểm khác đều xanh.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

_spec = importlib.util.spec_from_file_location(
    "ghep_ma_tran", GOC / "scripts" / "18_ghep_ma_tran.py")
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)


def _dung(thu_muc: Path, ten: str, **sidecar) -> Path:
    """Tạo một cặp `<ten>.npy` + `<ten>.json` giả. Không ghi `.npy` thật —
    `kiem_cung_model` chỉ đọc sidecar."""
    f = thu_muc / f"{ten}.npy"
    f.touch()
    if sidecar:
        f.with_suffix(".json").write_text(
            json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
    return f


GOPT = {"model": "ViT-gopt-16-SigLIP2-384", "pretrained": "webli", "chieu": 1536}


def test_cung_model_thi_qua(tmp_path):
    a = _dung(tmp_path, "chinh", **GOPT)
    b = _dung(tmp_path, "va", **GOPT)
    gm.kiem_cung_model(a, b)          # không ném gì là đạt


def test_khac_model_thi_dung(tmp_path):
    a = _dung(tmp_path, "chinh", **GOPT)
    b = _dung(tmp_path, "va", model="ViT-SO400M-14-SigLIP2-378",
              pretrained="webli", chieu=1152)
    with pytest.raises(SystemExit) as e:
        gm.kiem_cung_model(a, b)
    assert "KHÔNG CÙNG MODEL" in str(e.value)


def test_khac_pretrained_thi_dung(tmp_path):
    """Cùng model, khác trọng số — shape và dtype KHỚP HOÀN TOÀN.

    Đây là ca nguy hiểm nhất và là lý do hàm này tồn tại: không phép kiểm nào
    khác trong đường ống phân biệt được hai ma trận này.
    """
    a = _dung(tmp_path, "chinh", **GOPT)
    b = _dung(tmp_path, "va", model="ViT-gopt-16-SigLIP2-384",
              pretrained="/kaggle/input/mo-hinh-khac.safetensors", chieu=1536)
    with pytest.raises(SystemExit) as e:
        gm.kiem_cung_model(a, b)
    assert "pretrained" in str(e.value)


def test_thieu_sidecar_thi_canh_bao_chu_khong_chan(tmp_path, capsys):
    """Ma trận cũ có thể không có sidecar — không chặn, nhưng phải nói to."""
    a = _dung(tmp_path, "chinh", **GOPT)
    b = _dung(tmp_path, "va")                     # không sidecar
    gm.kiem_cung_model(a, b)
    ra = capsys.readouterr().out
    assert "KHÔNG CÓ sidecar" in ra
    assert "KHÔNG CÓ GÌ BÁO" in ra


def test_sidecar_thieu_khoa_thi_bo_qua_khoa_do(tmp_path):
    """Sidecar cũ thiếu `chieu` thì so hai khoá còn lại, đừng báo lệch nhầm."""
    a = _dung(tmp_path, "chinh", **GOPT)
    b = _dung(tmp_path, "va", model="ViT-gopt-16-SigLIP2-384",
              pretrained="webli")                 # không có `chieu`
    gm.kiem_cung_model(a, b)


def test_sidecar_hong_thi_coi_nhu_khong_co(tmp_path, capsys):
    a = _dung(tmp_path, "chinh", **GOPT)
    b = _dung(tmp_path, "va")
    b.with_suffix(".json").write_text("{ hong", encoding="utf-8")
    gm.kiem_cung_model(a, b)
    assert "KHÔNG CÓ sidecar" in capsys.readouterr().out
