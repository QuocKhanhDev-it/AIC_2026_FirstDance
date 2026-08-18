"""
test_sinh_caption.py — Chốt cho bộ sinh caption (kênh 5).

Chỉ kiểm phần KHÔNG gọi API: chọn tập, dọn văn bản, đọc lại tiến độ. Phần gọi
mạng không test được ở đây, nhưng ba thứ này mới là chỗ mất tiền nếu sai —
chọn nhầm tập là gọi thừa hàng nghìn lượt, đọc sai tiến độ là gọi lại từ đầu.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

_spec = importlib.util.spec_from_file_location(
    "sinh_caption", GOC / "scripts" / "14_sinh_caption.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def master_gia() -> pd.DataFrame:
    d = []
    for nhom, v, co_anh in (("L21", 0, True), ("L21", 1, False),
                            ("L22", 0, True), ("L27", 0, True)):
        for i in range(3):
            d.append({"row_id": len(d), "video_id": f"{nhom}_V{v:03d}",
                      "kf_n": i, "kf_name": f"{i:03d}.jpg",
                      "frame_idx": i * 25, "pts_time": float(i), "fps": 25.0,
                      "kf_path": f"/anh/{nhom}_{v}_{i}.jpg" if co_anh else None,
                      "title": "", "description": "", "keywords": ""})
    return pd.DataFrame(d)


def test_chi_chon_dong_co_anh():
    """Không có ảnh thì không có gì để nhìn. Mỗi máy giữ một phần kho (B4),
    nên gọi API cho dòng không ảnh là đốt quota vào hư không."""
    m = master_gia()
    d = sc.chon_row(m, "co-anh")
    assert len(d) == 9
    assert d.kf_path.notna().all()


def test_chon_theo_nhom():
    m = master_gia()
    assert set(sc.chon_row(m, "nhom:L21").video_id) == {"L21_V000"}
    assert len(sc.chon_row(m, "nhom:L21,L27")) == 6


def test_chon_sai_thi_dung_ngay():
    """Gõ sai `--chon` mà script lặng lẽ chạy cả kho là hóa đơn bất ngờ."""
    with pytest.raises(SystemExit):
        sc.chon_row(master_gia(), "toan-bo")


def test_don_bo_mo_dau_thua():
    """Chữ đưa đẩy có mặt ở 100% caption -> IDF ~ 0 nên vô hại về điểm, NHƯNG
    nó kéo dài tài liệu, mà BM25 phạt tài liệu dài. Chữ thừa hạ điểm mọi token
    thật."""
    assert sc.don("Bức ảnh cho thấy một người đàn ông") == "một người đàn ông"
    assert sc.don("Trong hình có: hai con mèo") == "hai con mèo"
    assert sc.don("  Một   người  phụ nữ ") == "Một người phụ nữ"


def test_don_khong_cat_cau_binh_thuong():
    t = "Hai người phụ nữ đang thái dứa trên thớt gỗ."
    assert sc.don(t) == t


def test_da_xong_chiu_duoc_dong_hong(tmp_path):
    """Ngắt giữa lúc ghi để lại một dòng cụt. Nếu `da_xong` chết vì dòng đó thì
    cả nghìn caption đã trả tiền coi như mất — phải bỏ qua dòng hỏng, giữ phần
    còn lại."""
    f = tmp_path / "caption.jsonl"
    f.write_text(
        json.dumps({"row_id": 1, "caption": "a"}, ensure_ascii=False) + "\n"
        + '{"row_id": 2, "capt\n'                       # dòng cụt do ngắt
        + json.dumps({"row_id": 3, "caption": "c"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    assert sc.da_xong(f) == {1, 3}


def test_da_xong_khi_chua_co_file(tmp_path):
    assert sc.da_xong(tmp_path / "chua_co.jsonl") == set()


def test_bien_cung_chiu_duoc_dong_hong(tmp_path):
    """`bien()` phải khoan dung y như `da_xong()`.

    Đã vấp: `da_xong()` bỏ qua dòng cụt nên lần chạy tiếp nối được, nhưng
    `bien()` dùng `pd.read_json(lines=True)` nên chết ở CUỐI — đúng lúc đã tiêu
    xong tiền API thì không lấy ra được `caption.parquet`.
    """
    f = tmp_path / "caption.jsonl"
    f.write_text(
        json.dumps({"row_id": 1, "caption": "một người"}, ensure_ascii=False) + "\n"
        + '{"row_id": 2, "capt\n'
        + json.dumps({"row_id": 3, "caption": "hai con mèo"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    sc.bien(f, tmp_path / "caption.parquet")
    d = pd.read_parquet(tmp_path / "caption.parquet")
    assert list(d.row_id) == [1, 3]
    assert d.caption.iloc[1] == "hai con mèo"


def test_bien_giu_ban_cuoi_khi_trung_row_id(tmp_path):
    """Chạy lại một `row_id` (vd sau khi sửa câu nhắc) phải đè bản cũ."""
    f = tmp_path / "caption.jsonl"
    f.write_text(
        json.dumps({"row_id": 7, "caption": "bản cũ"}, ensure_ascii=False) + "\n"
        + json.dumps({"row_id": 7, "caption": "bản mới"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    sc.bien(f, tmp_path / "caption.parquet")
    d = pd.read_parquet(tmp_path / "caption.parquet")
    assert len(d) == 1 and d.caption.iloc[0] == "bản mới"


def test_nhac_yeu_cau_tieng_viet():
    """Caption tiếng Anh + truy vấn tiếng Việt = khớp 0 token. BM25 khớp mặt
    chữ, không có không gian vector chung để bắc cầu như CLIP. Đây là đúng lỗi
    đã làm kênh 1 được 0,0000 (A10)."""
    assert "tiếng Việt" in sc.nhac()
    assert "tiếng Việt" in sc.nhac(dong_nghia=False)


def test_nut_dong_nghia_tat_bat_duoc():
    """Nới từ vựng phải A/B được: nó đánh đổi recall lấy độ dài tài liệu, mà
    BM25 phạt tài liệu dài. Chưa đo trên kho này nên không được chôn cứng."""
    bat, tat = sc.nhac(True), sc.nhac(False)
    assert len(bat) > len(tat)
    assert "khóm" in bat and "khóm" not in tat
