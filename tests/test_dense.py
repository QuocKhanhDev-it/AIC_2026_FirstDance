"""
test_dense.py — Chốt chặn cho bẫy A6.

Vì sao cần: sai biến thể model KHÔNG ném lỗi nào. Dùng `ViT-B-32` thay vì
`ViT-B-32-quickgelu` chỉ làm cosine tụt 0,9913 -> 0,9513 — hệ thống vẫn chạy,
vẫn trả kết quả, chỉ là kém đi một cách âm thầm. Không ai phát hiện ra cho tới
khi điểm dev thấp mà không hiểu vì sao.

`assert MODEL_TAG.endswith("-quickgelu")` không cứu được: nó chỉ kiểm chính
hằng số vừa gõ. Phép kiểm thật sự là chốt lại KẾT QUẢ: một truy vấn cố định
phải cho ra đúng `row_id` như đã ghi. Đổi model, đổi `pretrained`, nâng
`open_clip`, đổi cách chuẩn hóa — tất cả đều làm test đỏ.

Chạy:
    pytest tests/ -v

Cập nhật mốc (CHỈ khi cố ý đổi model, và phải ghi lý do vào commit):
    python src/dense.py --ghi-moc
"""

import json
import sys
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

MOC = Path(__file__).parent / "moc_dense.json"

pytestmark = pytest.mark.skipif(
    not (GOC / "index" / "clip.npy").exists(),
    reason="chưa có index/clip.npy trên máy này")


@pytest.fixture(scope="module")
def kenh():
    pytest.importorskip("open_clip", reason="cần requirements-clip.txt")
    from dense import KenhAnh
    return KenhAnh(GOC / "index")


@pytest.fixture(scope="module")
def moc():
    if not MOC.exists():
        pytest.skip(f"chưa có {MOC}. Chạy: python src/dense.py --ghi-moc")
    return json.loads(MOC.read_text("utf-8"))


def test_dung_bien_the_model(kenh, moc):
    """Bẫy A6: đổi model/pretrained là hỏng âm thầm."""
    assert kenh.model_tag == moc["model"]
    assert kenh.pretrained == moc["pretrained"]


def test_khong_doi_khong_gian_vector(kenh, moc):
    """Truy vấn cố định phải cho ra đúng khoảnh khắc như đã chốt.

    Đây là phép kiểm bắt được NHIỀU loại hỏng hơn cả: sai model, sai
    pretrained, quên chuẩn hóa L2, đổi thứ tự row_id, nâng thư viện làm đổi
    tokenizer.
    """
    dau = kenh.tim(moc["cau"], k=1)[0]
    assert dau.row_id == moc["row_id"], (
        f"top-1 đổi từ row_id {moc['row_id']} sang {dau.row_id}. "
        f"Nếu KHÔNG cố ý đổi model thì đây là lỗi — xem A6.")
    assert dau.video_id == moc["video_id"]
    assert abs(dau.score - moc["cos"]) < 5e-3


def test_vector_da_chuan_hoa(kenh):
    """`M @ q` chỉ là cosine khi cả hai đã chuẩn hóa L2."""
    import numpy as np
    v = kenh.encode_text("một câu bất kỳ")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    chuan = np.linalg.norm(np.asarray(kenh.mat[:200]), axis=1)
    assert np.allclose(chuan, 1.0, atol=1e-3), "ma trận chưa chuẩn hóa L2"


def test_master_va_ma_tran_cung_so_dong(kenh):
    """Lệch số dòng = mọi kết quả trỏ nhầm keyframe."""
    assert len(kenh.master) == kenh.mat.shape[0]


def test_so_chieu_doc_tu_ma_tran(kenh, moc):
    """Không hardcode 512 — để đổi sang SigLIP2 1152 chiều không phải sửa code."""
    assert kenh.chieu == kenh.mat.shape[1] == moc["chieu"]


def test_loc_theo_video(kenh):
    vid = kenh.master.video_id.iloc[100]
    kq = kenh.tim("người đi xe máy", k=20, video_id=vid)
    assert kq and all(c.video_id == vid for c in kq)


def test_khoa_be_ung_vien(kenh):
    """`be=` phải chặn thật, không được chỉ là tham số trang trí.

    Đây là chốt cho lỗi đo ở việc 8 của kế hoạch GPU: so `clip.npy` (đủ
    177.321 dòng thật) với ma trận chạy thử SigLIP2 (chỉ vài nghìn dòng thật)
    mà không khóa bể là so "tìm trong 177k" với "tìm trong vài nghìn".
    """
    import numpy as np
    vid = kenh.master.video_id.iloc[0]
    be = (kenh.master.video_id == vid).values

    kq = kenh.tim("người đi xe máy", k=50, be=be)
    assert kq, "khóa bể mà không trả về gì"
    assert all(c.video_id == vid for c in kq), "bể bị rò — có ứng viên ngoài bể"

    # thu hẹp bể phải làm thứ hạng của một đáp án cố định TỐT LÊN (số nhỏ hơn)
    from dense import be_chung
    assert be_chung(kenh).sum() == kenh.dong_da_encode().sum()
    assert isinstance(be_chung(kenh, kenh), np.ndarray)


def test_dong_da_encode(kenh):
    """`clip.npy` của BTC phủ 100% nên không được có dòng vector 0."""
    ma = kenh.dong_da_encode()
    assert ma.shape == (len(kenh.master),)
    assert ma.all(), f"{(~ma).sum()} dòng vector 0 trong clip.npy — bất thường"


def test_nhieu_bien_the_lay_diem_cao_nhat(kenh):
    """Nhiều cách diễn đạt -> max, không phải trung bình.

    Một cách diễn đạt trúng là đủ; không nên bị các cách diễn đạt trượt kéo
    xuống.
    """
    cau = "a person riding a motorbike on the street"
    mot = kenh.tim(cau, k=5)
    nhieu = kenh.tim([cau, "xyzzy khong lien quan gi ca"], k=5)
    assert nhieu[0].row_id == mot[0].row_id
    assert nhieu[0].score >= mot[0].score - 1e-6
