"""
test_nop_bai.py — Chốt cho bộ nộp bài.

Đây là chỗ **không có lần thử lại rẻ**: mỗi gói chỉ được nộp 3 lần, và *sai
định dạng vẫn tính là một lần nộp*. Một lỗi ở đây tốn 1/3 số lần nộp của cả
gói, chứ không phải "sửa rồi chạy lại".
"""

import csv
import io
import sys
import zipfile
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import nop_bai as nb                                      # noqa: E402
from schema import AnswerKIS, AnswerQA, AnswerTRAKE, Candidate  # noqa: E402


def kis(n=3):
    return [AnswerKIS(f"L01_V{i:03d}", 1000 + i) for i in range(n)]


# ---- tên gói ---------------------------------------------------------------

def test_ten_goi_sai_hau_to_bi_chan():
    """BTC chấm theo hậu tố tên file — `kis`/`qa`/`trake`. Đặt sai là bị chấm
    bằng bộ luật của loại khác."""
    assert nb.soat("ket-qua-1", kis())
    assert nb.soat("query-1-kiss", kis())
    assert not nb.soat("query-1-kis", kis())


def test_kieu_dap_an_phai_khop_hau_to():
    """File `-kis` mà chứa dòng Q&A là nộp thừa một cột, BTC parse hỏng."""
    loi = nb.soat("query-1-kis", [AnswerQA("L01_V001", 10, "5")])
    assert any("AnswerKIS" in x for x in loi), loi


# ---- giới hạn của BTC ------------------------------------------------------

def test_qua_100_dong_bi_chan():
    assert any("tối đa 100" in x for x in nb.soat("query-1-kis", kis(101)))
    assert not nb.soat("query-1-kis", kis(100))


def test_answer_qua_100_ky_tu_KHONG_bi_cat_lang_le():
    """Cắt bớt là ĐỔI câu trả lời thành một câu khác. Thà báo lỗi."""
    dai = AnswerQA("L01_V001", 10, "x" * 101)
    loi = nb.soat("query-2-qa", [dai])
    assert any("101 ký tự" in x for x in loi), loi
    # và phải KHÔNG ghi ra file nào
    with pytest.raises(SystemExit):
        nb.ghi_goi({"query-2-qa": [dai]}, "khong_duoc_tao")
    assert not Path("khong_duoc_tao").exists()


def test_answer_rong_bi_chan():
    assert nb.soat("query-2-qa", [AnswerQA("L01_V001", 10, "  ")])


def test_ten_video_con_duoi_mp4():
    """BTC ghi rõ `L01_V028` ✅ / `L01_V028.mp4` ❌."""
    loi = nb.soat("query-1-kis", [AnswerKIS("L01_V028.mp4", 10)])
    assert any(".mp4" in x for x in loi), loi


def test_frame_idx_phai_la_so_nguyen():
    assert nb.soat("query-1-kis", [AnswerKIS("L01_V001", 10.5)])
    assert nb.soat("query-1-kis", [AnswerKIS("L01_V001", -1)])


def test_dong_trung_bi_bao():
    """Chỉ có 100 chỗ; hai dòng y hệt phí một chỗ mà không tăng cơ hội nào."""
    d = AnswerKIS("L01_V001", 500)
    assert any("trùng" in x for x in nb.soat("query-1-kis", [d, d]))


# ---- TRAKE -----------------------------------------------------------------

def test_trake_dung_so_su_kien():
    ba = AnswerTRAKE("L10_V001", [100, 200, 300])
    assert not nb.soat("query-4-trake", [ba], so_su_kien=3)
    assert any("đòi 4 sự kiện" in x for x in
               nb.soat("query-4-trake", [ba], so_su_kien=4))


def test_trake_moi_dong_phai_cung_do_dai():
    loi = nb.soat("query-4-trake", [AnswerTRAKE("L10_V001", [1, 2, 3]),
                                    AnswerTRAKE("L10_V001", [1, 2])])
    assert any("dài khác nhau" in x for x in loi), loi


def test_trake_frame_phai_tang_dan():
    """*'Thứ tự các Frame ID phải tuân theo thứ tự thời gian của các events'*."""
    loi = nb.soat("query-4-trake", [AnswerTRAKE("L10_V001", [300, 100, 200])])
    assert any("tăng dần" in x for x in loi), loi


# ---- ghi file --------------------------------------------------------------

def test_ghi_khong_co_BOM(tmp_path):
    """`utf-8-sig` chèn EF BB BF vào đầu file -> tên video dòng đầu hỏng.

    Repo có chỗ đang dùng `utf-8-sig` (`05_bench_vlm.py`) nên đây không phải
    rủi ro lý thuyết.
    """
    nb.ghi_goi({"query-1-kis": kis()}, tmp_path / "s")
    tho = (tmp_path / "s" / "query-1-kis.csv").read_bytes()
    assert not tho.startswith(b"\xef\xbb\xbf")
    assert tho.startswith(b"L01_V000,")


def test_ghi_khong_co_header_va_dung_dinh_dang(tmp_path):
    nb.ghi_goi({"query-1-kis": [AnswerKIS("L00_V000", 1234)]}, tmp_path / "s")
    t = (tmp_path / "s" / "query-1-kis.csv").read_text("utf-8")
    assert t.replace("\r\n", "\n") == "L00_V000,1234\n"


def test_answer_co_dau_phay_duoc_boc_ngoac_kep(tmp_path):
    """Không bọc thì BTC parse thành hai cột -> answer bị cắt. Đây là lỗi BTC
    xếp trong năm lỗi thường gặp nhất."""
    nb.ghi_goi({"query-2-qa": [AnswerQA("L01_V028", 3450, "Có 3 người, cả nam và nữ")]},
               tmp_path / "s")
    t = (tmp_path / "s" / "query-2-qa.csv").read_text("utf-8")
    assert '"Có 3 người, cả nam và nữ"' in t
    hang = next(csv.reader(io.StringIO(t)))
    assert len(hang) == 3 and hang[2] == "Có 3 người, cả nam và nữ"


def test_answer_don_gian_khong_can_ngoac_kep(tmp_path):
    nb.ghi_goi({"query-2-qa": [AnswerQA("L01_V028", 3450, "5")]}, tmp_path / "s")
    assert '"' not in (tmp_path / "s" / "query-2-qa.csv").read_text("utf-8")


def test_answer_co_ngoac_kep_duoc_escape(tmp_path):
    nb.ghi_goi({"query-2-qa": [AnswerQA("L01_V028", 3450, 'Anh ấy nói "Xin chào"')]},
               tmp_path / "s")
    t = (tmp_path / "s" / "query-2-qa.csv").read_text("utf-8")
    assert next(csv.reader(io.StringIO(t)))[2] == 'Anh ấy nói "Xin chào"'


def test_mot_file_hong_thi_KHONG_ghi_file_nao(tmp_path):
    """Ghi một nửa rồi chết là tệ nhất: thư mục trông như đã xong."""
    d = tmp_path / "s"
    with pytest.raises(SystemExit):
        nb.ghi_goi({"query-1-kis": kis(),
                    "query-2-qa": [AnswerQA("L01_V001", 1, "")]}, d)
    assert not d.exists()


# ---- đóng gói --------------------------------------------------------------

def test_zip_co_thu_muc_submission_ben_trong(tmp_path):
    """*'KHÔNG nén trực tiếp các file CSV — phải nén thư mục submission'* —
    lỗi BTC xếp thứ hai trong năm lỗi thường gặp nhất."""
    d = nb.ghi_goi({"query-1-kis": kis()}, tmp_path / "s")
    z = nb.dong_goi(d, tmp_path / "b.zip")
    with zipfile.ZipFile(z) as f:
        ten = f.namelist()
    assert ten == ["submission/query-1-kis.csv"], ten


# ---- đọc ngược + cầu nối ---------------------------------------------------

def test_doc_lai_dung_nhu_da_ghi(tmp_path):
    goc = [AnswerQA("L01_V028", 3450, "Màu đỏ, rất đẹp")]
    nb.ghi_goi({"query-2-qa": goc}, tmp_path / "s")
    assert nb.doc_csv(tmp_path / "s" / "query-2-qa.csv") == goc


def test_doc_csv_bat_duoc_BOM(tmp_path):
    f = tmp_path / "query-1-kis.csv"
    f.write_bytes(b"\xef\xbb\xbfL01_V001,10\r\n")
    with pytest.raises(SystemExit, match="BOM"):
        nb.doc_csv(f)


def test_tu_ung_vien_lay_frame_idx_khong_tinh_lai():
    """`frame_idx` là giá trị nộp cho BTC — `Candidate` đã mang sẵn giá trị
    đúng từ bảng cái, ở đây chỉ đổi vỏ."""
    uv = [Candidate(row_id=5, video_id="L01_V001", frame_idx=26533,
                    score=0.9, meta={"pts_time": 1061.36})]
    ra = nb.tu_ung_vien(uv, "kis")
    assert ra == [AnswerKIS("L01_V001", 26533)]


def test_tu_ung_vien_qa_uu_tien_answer_rieng_tung_dong():
    """Mỗi dòng Q&A mang `answer` riêng (quy định BTC). `meta['answer']` thắng
    giá trị dùng chung."""
    uv = [Candidate(1, "L01_V001", 10, 0.9, meta={"answer": "bảy"}),
          Candidate(2, "L01_V002", 20, 0.8)]
    ra = nb.tu_ung_vien(uv, "qa", dap_an="năm")
    assert [x.answer for x in ra] == ["bảy", "năm"]


def test_tu_ung_vien_cat_dung_100_dong():
    uv = [Candidate(i, "L01_V001", i, 1.0) for i in range(150)]
    assert len(nb.tu_ung_vien(uv, "kis")) == 100


def test_tu_ung_vien_tu_choi_dung_TRAKE():
    """TRAKE mỗi dòng là một CHUỖI N khung, không dựng được từ danh sách phẳng."""
    with pytest.raises(ValueError, match="TRAKE"):
        nb.tu_ung_vien([Candidate(1, "L01_V001", 10, 1.0)], "trake")


def test_canh_bao_khi_chua_du_100_dong():
    """Không có điểm phạt — dòng thứ 100 vẫn đáng 0,2 nếu trúng."""
    assert nb.canh_bao("query-1-kis", kis(30))
    assert not nb.canh_bao("query-1-kis", kis(100))


# ---- soát chính file zip (checklist BTC 19/08/2026) -----------------------

def _zip_gia(tmp_path, cac_file: dict, ten="bainop.zip", tien_to="submission/"):
    import zipfile
    z = tmp_path / ten
    with zipfile.ZipFile(z, "w") as f:
        for n, noi_dung in cac_file.items():
            f.writestr(f"{tien_to}{n}" if tien_to else n, noi_dung)
    return z


def test_soat_zip_dat_khi_dung_chuan(tmp_path):
    z = _zip_gia(tmp_path, {"query-1-kis.csv": "L01_V028,25300\r\n"})
    loi, _ = nb.soat_zip(z)
    assert loi == []


def test_soat_zip_bat_thieu_thu_muc_submission(tmp_path):
    """Lỗi BTC xếp thứ hai trong năm lỗi thường gặp nhất."""
    z = _zip_gia(tmp_path, {"query-1-kis.csv": "L01_V028,25300\r\n"}, tien_to="")
    loi, _ = nb.soat_zip(z)
    assert any("submission" in x for x in loi)


def test_soat_zip_bat_file_khong_phai_csv(tmp_path):
    z = _zip_gia(tmp_path, {"query-1-kis.csv": "L01_V028,25300\r\n",
                            "ghi_chu.txt": "x"})
    loi, _ = nb.soat_zip(z)
    assert any(".csv" in x for x in loi)


def test_soat_zip_bat_khoang_trang_thua_theo_vi_du_trang_2_cua_btc(tmp_path):
    """BTC không trim, mà ví dụ trang 2 của chính họ lại có khoảng trắng.

    `L01_V028, 3450, "5"` đọc bằng parser CSV chuẩn ra answer `' "5"'` — khoảng
    trắng và ngoặc kép thành ký tự thật. Ai sửa tay theo ví dụ đó sẽ hỏng đúng
    dòng mình sửa mà nhìn không ra.
    """
    z = _zip_gia(tmp_path, {"query-2-qa.csv": 'L01_V028, 3450, "5"\r\n'})
    loi, _ = nb.soat_zip(z)
    assert any("khoảng trắng" in x for x in loi)


def test_soat_zip_bat_BOM(tmp_path):
    z = _zip_gia(tmp_path, {"query-1-kis.csv": "﻿L01_V028,25300\r\n"})
    loi, _ = nb.soat_zip(z)
    assert any("BOM" in x for x in loi)


def test_soat_zip_canh_bao_ten_zip_co_ky_tu_la(tmp_path):
    z = _zip_gia(tmp_path, {"query-1-kis.csv": "L01_V028,25300\r\n"},
                 ten="bai_nop_v2.zip")
    loi, canh = nb.soat_zip(z)
    assert loi == [] and any("khuyến cáo" in x for x in canh)


def test_soat_zip_van_ap_moi_luat_cua_soat(tmp_path):
    """Zip đúng cấu trúc nhưng nội dung sai vẫn phải bị chặn."""
    z = _zip_gia(tmp_path, {"query-4-trake.csv": "L01_V028,500,400\r\n"})
    loi, _ = nb.soat_zip(z)
    assert any("tăng dần" in x for x in loi)
