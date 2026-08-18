"""
test_bm25.py — Chốt cho bộ máy văn bản dùng chung của kênh 2, 3 và 5.

Kênh văn bản hỏng KHÔNG ném lỗi — nó trả về danh sách rỗng, hoặc tệ hơn, trả
về danh sách trông hợp lý mà xếp sai. Đúng kiểu hỏng đã làm kênh 1 được 0,0000
suốt một thời gian mà không ai biết.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from bm25 import BM25, KenhVanBan, bo_dau, don_metadata, tach   # noqa: E402


def master_gia(n_video=3, n_khung=4) -> pd.DataFrame:
    """Bảng cái tí hon, đủ hình dạng để kiểm logic mà không cần dữ liệu thật."""
    d = []
    for v in range(n_video):
        for i in range(n_khung):
            d.append({"row_id": len(d), "video_id": f"L99_V{v:03d}",
                      "kf_n": i, "kf_name": f"{i:03d}.jpg",
                      "frame_idx": i * 25, "pts_time": float(i),
                      "fps": 25.0, "kf_path": None,
                      "title": f"video {v}", "description": "", "keywords": ""})
    return pd.DataFrame(d)


# ---- tách token ------------------------------------------------------------

def test_bigram_khong_bac_qua_dau_cau():
    """`"xe máy. Hôm nay"` không được sinh `"máy_hôm"` — hai từ đó không đứng
    cạnh nhau về nghĩa, nối lại là bịa ra một cụm không có thật."""
    t = tach("xe máy. Hôm nay")
    assert "xe_máy" in t
    assert "hôm_nay" in t
    assert "máy_hôm" not in t


def test_bigram_lam_cum_hai_tu_hiem_hon():
    """Lý do có bigram: `"xe máy"` phải phân biệt được với `"máy xay"`."""
    b = BM25(["xe máy chạy trên đường", "máy xay sinh tố"])
    d = b.diem("xe máy")
    assert d[0] > d[1], "bigram không tách được 'xe máy' khỏi 'máy xay'"


def test_bo_dau_xu_rieng_chu_d():
    """`đ` là ký tự độc lập trong Unicode, `NFD` không tách ra được."""
    assert bo_dau("Đường") == "duong"
    assert bo_dau("cà chua") == "ca chua"


# ---- BM25 ------------------------------------------------------------------

def test_idf_khong_bao_gio_am():
    """Token có mặt ở quá nửa số tài liệu vẫn phải cho IDF >= 0.

    Dạng IDF cổ điển cho số ÂM trong trường hợp đó, biến token phổ biến thành
    hình phạt — tài liệu chứa từ khóa bị xếp DƯỚI tài liệu không chứa.
    """
    b = BM25(["mèo", "mèo", "mèo", "chó"])
    assert all(v >= 0 for v in b.idf.values()), b.idf
    d = b.diem("mèo")
    assert d[0] > d[3], "tài liệu chứa từ khóa bị xếp dưới tài liệu không chứa"


def test_khong_khop_thi_diem_khong():
    b = BM25(["mèo ngồi trên thảm"])
    assert b.diem("máy bay phản lực").sum() == 0.0


# ---- dấu tiếng Việt --------------------------------------------------------

def test_truy_van_khong_dau_van_tim_ra():
    """Người gõ nhanh trong phiên thi hay bỏ dấu; bài báo AIC'25 có truy vấn
    `"giai phong khi hidro"`. Không chuẩn hóa thì kênh im lặng trả 0."""
    m = master_gia(2)
    k = KenhVanBan(["giải phóng khí hidro trong ống nghiệm", "nấu canh chua cá"],
                   [np.array([0]), np.array([4])], m)
    d = k.diem_tai_lieu("giai phong khi hidro")
    assert d[0] > 0, "truy vấn không dấu không tìm ra tài liệu có dấu"
    assert d[0] > d[1]


def test_khop_dung_dau_xep_tren_khop_mo():
    """Cộng hai chỉ mục: khớp đúng dấu ăn điểm CẢ HAI, khớp mờ ăn MỘT."""
    m = master_gia(2)
    k = KenhVanBan(["con chó chạy", "cái chợ đông người"],
                   [np.array([0]), np.array([4])], m)
    d = k.diem_tai_lieu("chó")
    assert d[0] > d[1], "tài liệu khớp đúng dấu không được xếp trên"


# ---- kênh -> Candidate -----------------------------------------------------

def test_tim_lay_frame_idx_tu_master_khong_tu_tinh():
    """`frame_idx` là giá trị NỘP CHO BTC (schema.py). Tự tính lại từ pts_time
    lệch 1 frame vì làm tròn."""
    m = master_gia(1, 3)
    k = KenhVanBan(["mèo"], [np.array([0, 1, 2])], m)
    for c in k.tim("mèo", k=3):
        assert c.frame_idx == int(m.frame_idx.iloc[c.row_id])
        assert c.video_id == m.video_id.iloc[c.row_id]


def test_tim_khong_bia_them_ung_vien():
    """Hết tài liệu khớp thì DỪNG. Đệm cho đủ k bằng tài liệu điểm 0 là đưa
    rác vào RRF, mà RRF chỉ đọc thứ hạng nên rác đó ăn điểm thật."""
    m = master_gia(3)
    k = KenhVanBan(["mèo", "chó", "chim"],
                   [np.array([0]), np.array([4]), np.array([8])], m)
    assert len(k.tim("mèo", k=100)) == 1


def test_moi_video_gioi_han_dung():
    """Kênh cấp video đổ cả trăm khung của một video vào danh sách nếu không chặn."""
    m = master_gia(2, 5)
    k = KenhVanBan(["mèo", "mèo"],
                   [np.arange(0, 5), np.arange(5, 10)], m)
    assert len(k.tim("mèo", k=100, moi_video=2)) == 4
    assert len(k.tim("mèo", k=100)) == 10


def test_tu_bang_khung_bo_caption_rong():
    """Tài liệu rỗng làm lệch `dai_tb`, qua đó lệch điểm của MỌI tài liệu khác."""
    m = master_gia(1, 3)
    bang = pd.DataFrame({"row_id": [0, 1, 2],
                         "caption": ["một người đàn ông", "", None]})
    k = KenhVanBan.tu_bang_khung(m, bang)
    assert len(k) == 1


def test_don_metadata():
    """`don_metadata` phải xóa URL, hashtag và mention."""
    s = "Xem thêm tại https://youtube.com/watch?v=123 #amthuc #monngon @vtv3 để biết"
    sach = don_metadata(s)
    assert "https" not in sach
    assert "youtube" not in sach
    assert "#amthuc" not in sach
    assert "amthuc" not in sach
    assert "@vtv3" not in sach
    assert "vtv3" not in sach
    assert "Xem thêm tại" in sach
    assert "để biết" in sach


def test_don_metadata_khong_noi_lien_hai_ben_rac():
    """Rác nằm ở giữa câu khi xóa phải thay bằng '. ' để không sinh bigram lai hai bên."""
    s = "Chi tiết tại https://youtu.be/xyz123 #nauan @chef nấu ăn ngon nhé"
    sach = don_metadata(s)
    t = tach(sach)
    assert "tại_nấu" not in t, "Sinh spurious bigram 'tại_nấu' do nối liền 2 bên rác"
    assert "chi_tiết" in t
    assert "nấu_ăn" in t
    assert "ăn_ngon" in t


def test_tu_metadata_ngat_bigram_giua_cac_truong():
    """Ghép title + description + keywords bằng '. ' để không sinh bigram lai."""
    df = pd.DataFrame([{
        "row_id": 0, "video_id": "V001", "frame_idx": 0, "pts_time": 0.0,
        "fps": 25.0, "kf_n": 0, "kf_name": "0.jpg", "kf_path": None,
        "title": "ngày", "description": "món ngon", "keywords": "thực đơn"
    }])
    k = KenhVanBan.tu_metadata(df)
    # Lấy các token trong chỉ mục có dấu
    tokens = set(k.co_dau.chi_muc.keys())
    assert "ngày_món" not in tokens, "Sinh spurious bigram ngày_món tại biên title-description"
    assert "ngon_thực" not in tokens, "Sinh spurious bigram ngon_thực tại biên description-keywords"
    assert "ngày_ngày" not in tokens, "Sinh spurious bigram ngày_ngày giữa các lần lặp title"
    assert "món_ngon" in tokens
    assert "thực_đơn" in tokens


def test_don_metadata_khong_an_chu_that_sau_hashtag():
    """`#\\w+` chứ không phải `#\\S+`.

    `\\S+` chạy tới khoảng trắng gần nhất nên `"#amthuc,rau cu"` bị ăn luôn cả
    `rau`. Trên kho hiện tại chưa cắn (hashtag luôn có khoảng trắng theo sau),
    nhưng kênh 3 sắp đẩy chữ OCR qua đúng hàm này và chữ OCR bẩn hơn nhiều.
    """
    t = tach(don_metadata("món ngon #amthuc,rau củ xanh"))
    assert "rau" in t, "hashtag ăn luôn từ thật đứng sau dấu phẩy"
    assert "củ" in t
    assert "amthuc" not in t


def test_tu_metadata_don_ca_title():
    """Rác trong `title` ăn trọng số GẤP BA vì title lặp 3 lần.

    13/873 tiêu đề của kho có hashtag; bản vá đầu chỉ dọn description và
    keywords nên chỗ nặng nhất lại là chỗ bị bỏ sót.
    """
    df = pd.DataFrame([{
        "row_id": 0, "video_id": "V001", "frame_idx": 0, "pts_time": 0.0,
        "fps": 25.0, "kf_n": 0, "kf_name": "0.jpg", "kf_path": None,
        "title": "Múa lân #htvsports #htvthethao",
        "description": "", "keywords": "",
    }])
    tokens = set(KenhVanBan.tu_metadata(df).co_dau.chi_muc.keys())
    assert "htvsports" not in tokens
    assert "htvthethao" not in tokens
    assert "lân" in tokens


def test_tu_metadata_loc_rac_description_keywords():
    """URL, hashtag và mention trong metadata phải được dọn trước khi tạo BM25."""
    df = pd.DataFrame([{
        "row_id": 0, "video_id": "V001", "frame_idx": 0, "pts_time": 0.0,
        "fps": 25.0, "kf_n": 0, "kf_name": "0.jpg", "kf_path": None,
        "title": "Món ăn ngon",
        "description": "Chi tiết tại https://youtu.be/xyz123 #nauan @chef",
        "keywords": "cách nấu #monngon"
    }])
    k = KenhVanBan.tu_metadata(df)
    tokens = set(k.co_dau.chi_muc.keys())
    for rac in ("https", "youtu", "be", "xyz123", "nauan", "monngon", "chef"):
        assert rac not in tokens, f"Token rác '{rac}' vẫn lọt vào chỉ mục metadata"


# ---- dữ liệu thật ----------------------------------------------------------

@pytest.fixture(scope="module")
def master():
    p = GOC / "index" / "master.parquet"
    if not p.exists():
        pytest.skip("chưa có index/master.parquet")
    return pd.read_parquet(p)


def test_metadata_dung_mot_tai_lieu_moi_video(master):
    k = KenhVanBan.tu_metadata(master)
    assert len(k) == master.video_id.nunique()
    # khóa dòng phải phủ đúng toàn bộ keyframe, không thiếu không trùng
    tong = sum(len(x) for x in k.khoa_dong)
    assert tong == len(master)


def test_metadata_tim_ra_video_dung_tren_tap_dev(master):
    """Chốt hồi quy: nếu ai đó làm hỏng tách token hay công thức, số này tụt.

    Ngưỡng đặt thấp hơn kết quả đo được (94/97 tìm ra, 22 câu top-10) để không
    vỡ vì một câu dev mới, nhưng đủ chặt để bắt hỏng thật.
    """
    import tap_dev
    f = GOC / "dev" / "tap_dev.jsonl"
    if not f.exists():
        pytest.skip("chưa có tập dev")
    cau = tap_dev.doc(f)
    if len(cau) < 30:
        pytest.skip("tập dev quá nhỏ")

    k = KenhVanBan.tu_metadata(master)
    tim_ra = top10 = 0
    for c in cau:
        d = k.diem_tai_lieu(c.cau_hoi)
        xep = np.argsort(-d)
        dung = c.video_id(master)
        h = next((i + 1 for i, j in enumerate(xep)
                  if k.video_id[j] == dung and d[j] > 0), None)
        tim_ra += h is not None
        top10 += h is not None and h <= 10
    assert tim_ra / len(cau) > 0.85, f"chỉ tìm ra {tim_ra}/{len(cau)}"
    assert top10 / len(cau) > 0.12, f"chỉ {top10}/{len(cau)} câu có video ở top-10"


def test_kenh_5_noi_dung_vao_thuoc_do(master):
    """Kênh 5 chạy hết đường ống caption -> BM25 -> Candidate -> `cham()`.

    Chưa có caption thật (chờ khóa API), nên ở đây DỰNG caption bằng chính câu
    hỏi dev gắn vào khung đúng. Bài này KHÔNG nói gì về chất lượng VLM — nó chỉ
    chốt phần NỐI DÂY: nếu caption mô tả đúng cảnh thì kênh phải tìm ra.

    Có bài này thì hôm có khóa API, caption ra mà điểm vẫn 0, ta biết ngay lỗi
    ở caption chứ không phải ở đường ống.
    """
    import tap_dev
    from cham_diem import cham

    f = GOC / "dev" / "tap_dev.jsonl"
    if not f.exists():
        pytest.skip("chưa có tập dev")
    cau = tap_dev.doc(f)[:20]
    if len(cau) < 5:
        pytest.skip("tập dev quá nhỏ")

    bang = pd.DataFrame([{"row_id": int(c.row_id_dung[0]), "caption": c.cau_hoi}
                         for c in cau if c.loai != "TRAKE"])
    bang = bang.drop_duplicates("row_id")
    k5 = KenhVanBan.tu_bang_khung(master, bang, ten="caption")

    d = cham(cau, lambda c: k5.tim(c.cau_hoi, k=100))
    assert d.diem.mean() > 0.9, (
        f"đường ống kênh 5 đứt ở đâu đó: {d.diem.mean():.4f}\n"
        f"{d[d.diem < 1].to_string(index=False)}")
    assert d.iloc[0].loai and (d.hang == 1).sum() >= len(bang) - 2
