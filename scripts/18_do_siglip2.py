"""
18_do_siglip2.py — SigLIP2 toàn kho so với CLIP ViT-B/32, trên tập dev.

⛔ CẦN ≥ 16 GB RAM. ĐỪNG CHẠY TRÊN MÁY YẾU.
============================================
`ViT-SO400M-14-SigLIP2-378` chiếm ~3,5 GB trọng số, cộng `clip_siglip2.npy`
390 MB và `master.parquet`. Đã **làm treo/crash một máy 7,7 GB nhiều lần** trước
khi có bản vá `_nhan()` theo lô — và ngay cả sau bản vá thì phần nạp model vẫn
sát trần. Kết quả đã đo xong và chép nguyên vào **A17**; không cần chạy lại để
đọc con số.

Muốn chạy lại thì để dành cho **máy GPU của Khánh**, hoặc bất cứ máy nào ≥ 16 GB.

Đây là phép đo A10.3 hứa hẹn, làm lại ở quy mô thật. A10.3 chỉ chạy được trên
11 video / 3.135 keyframe của L21+L22 vì chưa có ma trận toàn kho, và đã ghi
kèm cảnh báo **đừng đọc con số tuyệt đối 0,86–0,94 là năng lực hệ thống**.
Nay có `clip_siglip2.npy` đủ 177.321 dòng nên con số đọc thẳng được.

BỂ ỨNG VIÊN — LẦN NÀY KHÔNG CẦN KHÓA, VÀ ĐÓ LÀ ĐIỂM MẤU CHỐT
Cả hai ma trận đều phủ 100% kho, nên `be_chung()` trả về đúng toàn bộ và không
còn hiệu ứng thổi phồng +0,2833 đã đo ở `dense.be_chung`. Script vẫn tính bể
chung và **in ra kích thước** — nếu con số đó nhỏ hơn 177.321 thì có ma trận nào
đó chưa đầy, và mọi so sánh bên dưới lại phải đọc như A10.3.

CHẠY TỪNG MODEL MỘT, KHÔNG GIỮ CẢ HAI TRONG RAM
`ViT-SO400M-14-SigLIP2-378` chiếm ~3,5 GB. Máy 7,7 GB không ôm nổi nó cùng lúc
với CLIP và hai ma trận. Nên script quét xong một model, **giải phóng**, rồi mới
nạp model kia — chỉ giữ lại `list[Candidate]`, thứ nhẹ hều.

    python scripts/18_do_siglip2.py
    python scripts/18_do_siglip2.py --moi-video 3
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np


GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from cham_diem import bao_cao_do_nhay, MOC_DUNG_SAI   # noqa: E402
from dense import KenhAnh                             # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def co_vector(npy: Path) -> np.ndarray:
    """Mặt nạ dòng đã encode thật — đọc THẲNG từ `.npy`, không nạp model.

    `dense.be_chung()` cần dựng `KenhAnh`, mà dựng `KenhAnh` là nạp model. Ở
    đây chỉ cần biết dòng nào khác 0, nên đọc thẳng ma trận rẻ hơn nhiều.
    """
    m = np.load(npy, mmap_mode="r")
    ra = np.zeros(m.shape[0], dtype=bool)
    for i in range(0, m.shape[0], 20000):
        j = min(i + 20000, m.shape[0])
        ra[i:j] = np.abs(np.asarray(m[i:j, :8], np.float32)).sum(1) > 0
    return ra


def quet(index: Path, matrix: str, cau, k: int, mv, be) -> dict:
    """Quét cả tập dev bằng MỘT model, rồi giải phóng model đó.

    Trả `{id_câu: list[Candidate]}` — vài nghìn dataclass, không đáng kể so với
    3,5 GB trọng số.
    """
    kenh = KenhAnh(index, matrix=matrix, mmap=True)
    print(f"  {matrix:<22} {kenh.mat.shape}  {kenh.model_tag}/{kenh.pretrained}")
    ra = {c.id: kenh.tim(c.cau_hoi, k=k, be=be, moi_video=mv) for c in cau}
    master = kenh.master
    del kenh
    gc.collect()
    return ra, master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--moi-video", type=int, default=0)
    a = ap.parse_args()

    cau = tap_dev.doc(a.file)
    mv = a.moi_video or None

    be = co_vector(a.index / "clip.npy") & co_vector(a.index / "clip_siglip2.npy")
    day, tong = int(be.sum()), len(be)
    print(f"bể chung: {day:,}/{tong:,} keyframe")
    if day < tong:
        print("⚠️  Bể KHÔNG đầy — có ma trận chưa encode hết. Con số tuyệt đối\n"
              "    bên dưới bị thổi phồng (xem be_chung), chỉ đọc phần SO SÁNH.")
    else:
        print("   Đầy đủ -> con số tuyệt đối đọc thẳng được, không như A10.3.")
        be = None                      # bể đầy thì khỏi nhân thêm mặt nạ
    print(f"{len(cau)} câu | k={a.k} | moi_video={mv or 'tắt'}\n")

    kq32, master = quet(a.index, "clip.npy", cau, a.k, mv, be)
    kqsig, _ = quet(a.index, "clip_siglip2.npy", cau, a.k, mv, be)

    # Kênh văn bản + objects: nhẹ, nạp sau khi đã giải phóng hai model ảnh.
    from bm25 import KenhVanBan
    k2 = KenhVanBan.tu_metadata(master)
    kq2 = {c.id: k2.tim(c.cau_hoi, k=a.k, moi_video=3) for c in cau}
    del k2
    gc.collect()

    sys.path.insert(0, str(GOC / "scripts"))
    import importlib.util
    s = importlib.util.spec_from_file_location("r16", GOC / "scripts" / "16_do_rrf.py")
    r16 = importlib.util.module_from_spec(s)
    s.loader.exec_module(r16)
    k4 = r16.KenhObjects(a.index, master)
    kq4 = {c.id: k4.tim(c.cau_hoi, k=a.k) for c in cau}
    del k4
    gc.collect()

    # ⚠️ MỐC NỀN LÀ SigLIP2 — cấu hình MẠNH NHẤT hiện có, không phải CLIP.
    # So với CLIP (đang 0,0000) thì mọi thứ đều thắng, mà thắng vậy chẳng nói
    # lên điều gì. Câu hỏi thật là: có gì cộng thêm vào SigLIP2 mà LÃI không?
    print()
    print(bao_cao_do_nhay(cau, {
        "SigLIP2 SO400M (mốc)": lambda c: kqsig[c.id],
        "CLIP ViT-B/32": lambda c: kq32[c.id],
        "RRF(SigLIP2, CLIP)": lambda c: hop_nhat([kqsig[c.id], kq32[c.id]]),
        "RRF(SigLIP2, objects)": lambda c: hop_nhat([kqsig[c.id], kq4[c.id]]),
        "RRF(SigLIP2, metadata)": lambda c: hop_nhat([kqsig[c.id], kq2[c.id]]),
        "RRF(cả bốn kênh)": lambda c: hop_nhat(
            [kqsig[c.id], kq32[c.id], kq4[c.id], kq2[c.id]]),
    }, master=master, moc=MOC_DUNG_SAI, gioi_han=a.k))


if __name__ == "__main__":
    main()
