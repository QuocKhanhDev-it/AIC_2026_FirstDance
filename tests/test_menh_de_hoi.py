"""
test_menh_de_hoi.py — Chốt cho bộ nhận diện mệnh đề HỎI (A93).

Bộ nhận diện này quyết định câu nào bị `--trong-so-hoi` đụng tới. Nó bắn nhầm
vào một câu KIS là đổi kết quả của câu đó mà không ai đo; bắn thiếu ở câu Q&A
là cờ không làm gì. Cả hai đều im lặng, nên phải có chốt.
"""

import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))


def test_bat_dung_menh_de_hoi_va_tha_menh_de_ta_canh():
    from run import la_menh_de_hoi
    hoi = [
        "Mỗi lần khuôn này làm được bao nhiêu cái bánh?",
        "Con số được ghi trên biển báo bên trái của cây cầu là bao nhiêu?",
        "Hỏi quán trọ được nhắc đến trong đoạn phim nằm trên đường nào?",
        # dạng mệnh lệnh, KHÔNG có dấu hỏi — `?` một mình là không đủ
        "Hãy cho biết loài cây (II) đạt tốc độ sinh trưởng tốt nhất khi nào",
    ]
    ta = [
        "Đoạn video mô tả quá trình làm bánh, bánh được tạo ra có màu tím.",
        "Một người đàn ông mặc áo đỏ đang phát biểu trên sân khấu lớn.",
        "Đoạn clip mô tả hậu quả của hiện tượng sạt lở đất nghiêm trọng.",
    ]
    for m in hoi:
        assert la_menh_de_hoi(m), f"bỏ sót mệnh đề hỏi: {m}"
    for m in ta:
        assert not la_menh_de_hoi(m), f"bắn nhầm vào mệnh đề tả cảnh: {m}"


def test_cau_khong_dau_van_bat_duoc_neu_co_dau_hoi():
    """Truy vấn gõ thiếu dấu vẫn phải bắt được — miễn là còn dấu `?`.

    ⚠️ HẠN CHẾ ĐÃ BIẾT: câu vừa KHÔNG dấu vừa KHÔNG có `?` thì lọt. Danh sách
    từ khoá khớp bản CÓ DẤU, và cố ý giữ vậy: đổi sang khớp bản bỏ dấu sẽ đổi
    tập câu bị ảnh hưởng, làm phép đo A93 không còn tái lập được.
    """
    from run import la_menh_de_hoi
    assert la_menh_de_hoi("Hoi quan tro duoc nhac den nam tren duong nao?")
    assert not la_menh_de_hoi("Hay cho biet con so tren bien bao la bao nhieu")


def test_trong_so_hoi_mac_dinh_1_khong_doi_gi():
    """Mặc định phải là hành vi CŨ y hệt — A93 mới 🟡, chưa được bật."""
    import argparse
    import run as R
    ap = argparse.ArgumentParser()
    for h in (R.them_tham_so,) if hasattr(R, "them_tham_so") else ():
        h(ap)
    # Không dựng lại parser thì đọc thẳng mặc định đã khai báo.
    import re
    nguon = (GOC / "src" / "run.py").read_text("utf-8")
    m = re.search(r'"--trong-so-hoi",\s*type=float,\s*default=([\d.]+)', nguon)
    assert m, "không tìm thấy khai báo --trong-so-hoi"
    assert float(m.group(1)) == 1.0, (
        "A93 mới đạt 🟡 (+0,0154, dưới ngưỡng 0,0287) nên KHÔNG được bật mặc "
        "định. Muốn bật thì phải có phép đo vượt ngưỡng trước.")
