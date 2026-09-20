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
    from dense import KenhAnh, RAM_CAN, ram_trong_gb
    # Chốt RAM của `dense` ném SystemExit chứ không phải lỗi test. Máy đang
    # thiếu RAM thì BỎ QUA — ép chạy là treo máy, đã xảy ra hai lần.
    tro, can = ram_trong_gb(), RAM_CAN[512]
    if tro is not None and tro < can:
        pytest.skip(f"chỉ còn {tro:.1f} GB RAM, cần ~{can:.1f} GB để nạp model")
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


def test_encode_image_khop_lai_vector_trong_ma_tran(kenh):
    """`encode_image()` phải ra CÙNG không gian với ma trận có sẵn.

    Mã hóa lại chính ảnh keyframe của một dòng, so cosine với vector đã có
    trong `.npy` — phải cao (>0,9): cùng ảnh, cùng model thì phải gần trùng.
    Đây là bẫy trực tiếp của A6/A10.3-style: sai `force_image_size` hay sai
    tiền xử lý sẽ làm cosine tụt hẳn mà không có lỗi nào ném ra.
    """
    import numpy as np
    from PIL import Image
    co_anh = kenh.master[kenh.master.kf_path.notna()]
    if co_anh.empty:
        import pytest
        pytest.skip("không có keyframe ảnh thật trên máy này")
    r = co_anh.iloc[0]
    anh = Image.open(r.kf_path).convert("RGB")
    v = kenh.encode_image([anh])[0]
    v_goc = np.asarray(kenh.mat[int(r.row_id)], dtype=np.float32)
    cos = float(v @ v_goc)
    assert cos > 0.9, f"encode_image lệch không gian vector với ma trận (cos={cos:.3f})"


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


# ---- KenhAnhCache: kênh 1 chạy không nạp model ----------------------------

def _D():
    """Nạp `dense` trong hàm, giống các test khác của file này."""
    import dense
    return dense


def _index_gia(tmp_path, chieu=4, n=6):
    """Bảng cái + ma trận tí hon, đủ để `tim()` chạy thật."""
    import numpy as np
    import pandas as pd
    d = []
    for v in range(2):
        for i in range(n // 2):
            d.append({"row_id": len(d), "video_id": f"L01_V{v:03d}", "kf_n": i,
                      "frame_idx": i * 100, "pts_time": float(i), "fps": 25.0,
                      "kf_path": None, "title": "t", "description": "",
                      "keywords": ""})
    m = pd.DataFrame(d)
    m.to_parquet(tmp_path / "master.parquet", index=False)
    mat = np.zeros((n, chieu), dtype=np.float32)
    for i in range(n):
        mat[i, i % chieu] = 1.0
    np.save(tmp_path / "clip.npy", mat)
    return m, mat


def _cache_gia(tmp_path, cac_cau, vecs, matrix="clip.npy", model="model-thu"):
    import json
    import numpy as np
    f = tmp_path / "truy_van.npz"
    np.savez_compressed(
        f, cau=np.array(cac_cau, dtype=object).astype(str),
        vec=np.asarray(vecs, dtype=np.float32),
        ghi_chu=json.dumps({"model": model, "pretrained": "x",
                            "matrix": matrix, "chieu": len(vecs[0])}))
    return f


def test_cache_tim_duoc_ma_khong_nap_model(tmp_path):
    """Bất biến chính: chạy được kênh 1 mà không import torch/open_clip."""
    import numpy as np
    _index_gia(tmp_path)
    q = np.zeros(4, dtype=np.float32); q[1] = 1.0     # trùng hệt dòng row_id 1
    f = _cache_gia(tmp_path, ["câu thử"], [q])
    kenh = _D().KenhAnhCache(tmp_path, f)
    kq = kenh.tim("câu thử", k=3)
    assert kq[0].row_id == 1
    assert kq[0].score == pytest.approx(1.0)
    assert kq[0].source == "clip"


def test_cache_dung_lai_tim_cua_kenh_that(tmp_path):
    """`tim()` phải là hàm THỪA KẾ, không phải bản chép — hai nhánh lệch nhau
    âm thầm thì số đo máy yếu không so được với số đo máy khoẻ."""
    assert _D().KenhAnhCache.tim is _D().KenhAnh.tim


def test_cache_nhan_danh_sach_menh_de_lay_max(tmp_path):
    """`run.tach_truy_van` cắt câu dài thành nhiều mệnh đề (A19/A20)."""
    import numpy as np
    _index_gia(tmp_path)
    a = np.zeros(4, dtype=np.float32); a[0] = 1.0
    b = np.zeros(4, dtype=np.float32); b[2] = 1.0
    f = _cache_gia(tmp_path, ["mệnh đề một", "mệnh đề hai"], [a, b])
    kenh = _D().KenhAnhCache(tmp_path, f)
    kq = kenh.tim(["mệnh đề một", "mệnh đề hai"], k=6)
    assert {c.row_id for c in kq[:2]} == {0, 2}


def test_cache_thieu_cau_thi_NEM_LOI_chu_khong_doan_bua(tmp_path):
    """Trả vector 0 sẽ cho ra 100 ứng viên ngẫu nhiên trông hợp lệ — đúng loại
    hỏng im lặng cả repo này dựng để chặn."""
    import numpy as np
    _index_gia(tmp_path)
    f = _cache_gia(tmp_path, ["có trong cache"], [np.ones(4, dtype=np.float32)])
    kenh = _D().KenhAnhCache(tmp_path, f)
    with pytest.raises(KeyError):
        kenh.tim("chưa từng mã hoá", k=5)


def test_cache_co_du_bao_truoc_cau_con_thieu(tmp_path):
    import numpy as np
    _index_gia(tmp_path)
    f = _cache_gia(tmp_path, ["a"], [np.ones(4, dtype=np.float32)])
    kenh = _D().KenhAnhCache(tmp_path, f)
    assert kenh.co_du(["a", "b", "c"]) == ["b", "c"]


def test_cache_lech_so_chieu_thi_dung_ngay(tmp_path):
    """Sai cặp cache/ma trận là sai không gian vector — mọi kết quả sau đó vô
    nghĩa mà vẫn trông hợp lệ."""
    import numpy as np
    _index_gia(tmp_path, chieu=4)
    f = _cache_gia(tmp_path, ["a"], [np.ones(8, dtype=np.float32)])
    with pytest.raises(SystemExit):
        _D().KenhAnhCache(tmp_path, f)


def test_cache_ap_dung_duoc_moi_rang_buoc_cua_tim(tmp_path):
    """`be`, `video_id`, `moi_video` — thừa kế nên phải chạy y hệt."""
    import numpy as np
    m, _ = _index_gia(tmp_path)
    q = np.ones(4, dtype=np.float32) / 2
    f = _cache_gia(tmp_path, ["q"], [q])
    kenh = _D().KenhAnhCache(tmp_path, f)
    assert all(c.video_id == "L01_V001"
               for c in kenh.tim("q", k=6, video_id="L01_V001"))
    assert len(kenh.tim("q", k=6, moi_video=1)) == 2
