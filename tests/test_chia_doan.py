"""
test_chia_doan.py — Chốt cho bước chia đoạn của kênh 6 (`scripts/55_`).

Ba thứ đáng chốt, và cả ba đều là loại hỏng KHÔNG ném lỗi:

  * đoạn dài quá trần 64 token -> tháp văn bản lặng lẽ cắt cụt (A51)
  * đoạn không phủ hết tài liệu -> mất chữ mà không ai biết
  * tokenizer bị gọi từng từ một -> đúng, nhưng chậm tới mức không dùng được;
    một lượt Kaggle 4 tiếng không qua nổi bước này. CHẬM cũng là một loại hỏng.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

_spec = importlib.util.spec_from_file_location(
    "ma_hoa_van_ban", GOC / "scripts" / "55_ma_hoa_van_ban.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

CTX = 64
THEM = 2          # token mở/đóng, chuỗi nào cũng có — giống tokenizer thật


class TokGia:
    """Tokenizer giả: 1 token cho mỗi 3 ký tự của từ, cộng 2 token bao ngoài.

    Đếm luôn số lần bị gọi — đó là thứ phép đo tốc độ thật sự nhìn vào.
    """

    def __init__(self):
        self.so_lan_goi = 0

    @staticmethod
    def _dem(s: str) -> int:
        return THEM + sum(-(-len(t) // 3) for t in s.split())

    def __call__(self, cac_chuoi):
        self.so_lan_goi += 1
        m = np.zeros((len(cac_chuoi), CTX), dtype=np.int64)
        for i, s in enumerate(cac_chuoi):
            m[i, :min(self._dem(s), CTX)] = 1
        return m


@pytest.fixture
def tok():
    return TokGia()


def _tai_lieu(n_tu: int, dai: int = 7) -> str:
    return " ".join(f"tu{i:0{dai}d}" for i in range(n_tu))


def test_moi_doan_nam_duoi_tran(tok):
    """Điều kiện sống còn của kênh 6: không đoạn nào bị tháp văn bản cắt cụt."""
    van = [_tai_lieu(300), _tai_lieu(5), _tai_lieu(64)]
    nhom = M.chia_doan_hang_loat(van, tok, 60, moi_lan_in=0)
    for ds in nhom:
        for doan in ds:
            assert tok._dem(doan) <= CTX


def test_khong_mat_chu(tok):
    """Ghép các đoạn lại phải ra đúng tài liệu gốc — không rơi, không lặp."""
    van = [_tai_lieu(300), _tai_lieu(41)]
    nhom = M.chia_doan_hang_loat(van, tok, 60, moi_lan_in=0)
    for goc, ds in zip(van, nhom):
        assert " ".join(ds) == goc


def test_tai_lieu_rong_khong_sinh_doan(tok):
    assert M.chia_doan_hang_loat(["", "   "], tok, 60, moi_lan_in=0) == [[], []]


def test_token_hoa_theo_lo_chu_khong_theo_tung_tu(tok):
    """Chốt chống hồi quy cho sự cố 4 tiếng.

    2.000 tài liệu × 100 từ, nhưng chỉ 100 từ DUY NHẤT. Bản cũ gọi tokenizer
    200.000 lần; bản này chỉ được phép gọi vài lần (một lần đo chuỗi rỗng,
    một ít lô từ vựng).
    """
    van = [_tai_lieu(100)] * 2000
    M.chia_doan_hang_loat(van, tok, 60, moi_lan_in=0)
    assert tok.so_lan_goi <= 5


def test_soat_tran_bat_duoc_doan_qua_dai(tok):
    """`soat_tran` là lưới an toàn — nó phải thật sự bắt được cá."""
    ngan, dai = _tai_lieu(3), _tai_lieu(200)
    assert M.soat_tran([ngan, dai, ngan], tok) == [1]
    assert M.soat_tran([ngan], tok) == []


def test_dem_token_tru_phan_bao_ngoai(tok):
    """Token của các từ phải CỘNG ĐƯỢC với nhau, nên phải trừ phần chung."""
    assert M.dem_token(["tu00000"], tok) == [3]         # 7 ký tự -> 3 token
    assert M.dem_token([""], tok) == [1]                # sàn 1, không phải 0
