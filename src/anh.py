"""
anh.py — Tìm ảnh của một keyframe, tự lùi về bản thu nhỏ khi không có ảnh gốc.

VÌ SAO TỒN TẠI

`kf_path` nghĩa là "ảnh đã tải **ở máy này**", không phải "có keyframe" (A5.5).
Mỗi máy giữ một phần kho khác nhau, và **L26 — 45% kho — thì không máy nào
trong nhóm có**: 12,13 GB, quá lớn để ai cũng giữ một bản.

Nhưng truy hồi không cần ảnh gốc. Ảnh chỉ cần cho hai việc — người soi bằng
mắt, và trả lời câu Q&A — mà cả hai chỉ cần **nhận ra cảnh**. Bản thu nhỏ 256px
(`scripts/49_sinh_anh_nho.py`) đo được **7,9 KB/ảnh**, tức 4,0% ảnh gốc: cả kho
177.321 ảnh gói lại còn **1,34 GB**, vừa mọi máy.

Module này là chỗ DUY NHẤT biết luật lùi đó. `web/server.py`, `src/tac_tu.py`
và mọi thứ khác gọi vào đây thay vì tự ghép đường dẫn — thêm một nguồn ảnh sau
này chỉ phải sửa một hàm.

⚠️ THU NHỎ KHÔNG PHẢI LÚC NÀO CŨNG DÙNG ĐƯỢC. Nó đủ để **nhận ra cảnh**, không
đủ để **đọc chữ nhỏ** trên biển hiệu hay bảng điểm — mà phần lớn câu Q&A của đề
thật lại là câu đọc chữ (10/14 câu, xem docs/15). Vì vậy `tim()` trả về cả cờ
`la_ban_nho`, và nơi gọi phải nói cho người dùng biết họ đang nhìn bản nào.
"""

from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
MAC_DINH_NHO = GOC / "index" / "anh_nho"


def duong_ban_nho(video_id: str, kf_n, goc=MAC_DINH_NHO) -> Path:
    """Đường dẫn bản thu nhỏ. Cấu trúc CỐ Ý giống hệt cây keyframe gốc."""
    return Path(goc) / str(video_id) / f"{int(kf_n):03d}.jpg"


def tim(kf_path, video_id: str, kf_n, goc=MAC_DINH_NHO) -> tuple[Path | None, bool]:
    """Trả về `(đường dẫn, la_ban_nho)`. `(None, False)` nếu không có ảnh nào.

    Ưu tiên ảnh gốc: nó nét hơn, và đọc được chữ nhỏ. Chỉ lùi về bản thu nhỏ
    khi ảnh gốc không có trên máy này.
    """
    if isinstance(kf_path, str) and kf_path:
        p = Path(kf_path)
        if p.exists():
            return p, False
    nho = duong_ban_nho(video_id, kf_n, goc)
    if nho.exists():
        return nho, True
    return None, False


def thong_ke(master, goc=MAC_DINH_NHO) -> dict:
    """Đếm xem soi được bao nhiêu dòng, và nhờ nguồn nào.

    Dùng để trả lời đúng một câu: *"máy này soi được bao nhiêu phần kho?"* —
    câu mà trước đây phải suy từ `kf_path.notna()` và luôn ra thiếu.
    """
    goc = Path(goc)
    co_goc = int(master.kf_path.notna().sum())
    if not goc.is_dir():
        return {"anh_goc": co_goc, "ban_nho": 0, "tong_soi_duoc": co_goc,
                "tong_dong": len(master)}

    # Đếm theo THƯ MỤC VIDEO, không stat từng dòng: 177.321 lần `exists()` trên
    # Windows mất hàng chục giây, mà bản thu nhỏ sinh trọn vẹn theo video nên
    # đếm file trong thư mục là đủ chính xác.
    co = {d.name: len(list(d.glob("*.jpg"))) for d in goc.iterdir() if d.is_dir()}
    thieu_goc = master[master.kf_path.isna()]
    nho = int(sum(min(n, int((thieu_goc.video_id == v).sum()))
                  for v, n in co.items()))
    return {"anh_goc": co_goc, "ban_nho": nho,
            "tong_soi_duoc": co_goc + nho, "tong_dong": len(master)}
