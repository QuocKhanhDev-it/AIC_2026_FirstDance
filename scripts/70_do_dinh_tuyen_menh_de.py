"""
70_do_dinh_tuyen_menh_de.py — Đưa mệnh đề vào ĐÚNG kênh, thay vì ném hết vào tất cả.

    python scripts/70_do_dinh_tuyen_menh_de.py

VẤN ĐỀ

Hiện mọi mệnh đề đều đi qua cả kênh 1 (ảnh) lẫn kênh 3 (BM25 trên OCR+ASR gộp
chung). Mệnh đề tả hình ảnh thuần ("người phụ nữ áo đỏ đứng cạnh xe máy") chạy
qua BM25 chỉ sinh điểm rác; ngược lại mệnh đề nói về CHỮ TRÊN MÀN HÌNH ("biển
hiệu ghi BỆNH VIỆN") thì tháp ảnh của SigLIP2 đọc rất kém.

BỐN CẤU HÌNH, TÁCH HAI THAY ĐỔI KHÁC NHAU

Kênh 3 hiện là MỘT BM25 trên cột `text` = OCR + ASR nối lại. Bảng `ocr_asr` có
sẵn `ocr_text` và `asr_text` riêng, nên có thể tách thành hai kênh. Nhưng
"tách kênh" và "định tuyến mệnh đề" là HAI thay đổi:

  2. định tuyến, kênh 3 vẫn gộp  — chỉ đổi thứ ĐƯA VÀO
  3. tách hai kênh, KHÔNG định tuyến (cả hai nhận cả câu) — chỉ đổi CẤU TRÚC
  4. tách hai kênh + định tuyến — cả hai

Không có dòng 3 thì một cải thiện ở dòng 4 sẽ bị quy nhầm cho định tuyến.

⚠️ Bộ luật nhận diện là regex từ khoá, không phải model. Nó sẽ nhận nhầm. Vì
vậy script in ra SỐ MỆNH ĐỀ mỗi loại trước khi đo — nếu chỉ vài mệnh đề bị
định tuyến khác đi thì kết quả nằm dưới ngưỡng nhiễu theo thiết kế, và đó là
kết luận chứ không phải thất bại của ý tưởng.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5

# Mệnh đề nói về CHỮ HIỂN THỊ trên khung hình -> kênh OCR
CHU = re.compile(
    r"\b(chữ|dòng chữ|hàng chữ|biển|bảng|băng.?rôn|khẩu hiệu|tiêu đề|nhãn|"
    r"logo|ghi|đề|biển số|tên gọi|dòng tít|phụ đề|chú thích)\b", re.I)
# Mệnh đề nói về LỜI NÓI -> kênh ASR
LOI = re.compile(
    r"\b(nói|phát biểu|cho biết|kể|giới thiệu|trả lời|phỏng vấn|chia sẻ|"
    r"khẳng định|bình luận|tuyên bố|thuyết minh|dẫn chuyện|lời)\b", re.I)


def phan_loai(md: list[str]) -> tuple[list, list, list]:
    """-> (mệnh đề chữ, mệnh đề lời, mệnh đề thị giác)."""
    chu = [m for m in md if CHU.search(m)]
    loi = [m for m in md if LOI.search(m)]
    thi_giac = [m for m in md if m not in chu and m not in loi]
    return chu, loi, thi_giac


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")
    k_ocr = KenhVanBan.tu_bang_khung(master, bang, cot="ocr_text", ten="ocr")
    k_asr = KenhVanBan.tu_bang_khung(master, bang, cot="asr_text", ten="asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu\n")

    # Có gì để định tuyến không? In TRƯỚC khi đo.
    n_chu = n_loi = n_tg = 0
    co_chu = co_loi = 0
    for c in giu:
        chu, loi, tg = phan_loai(R.tach_truy_van(c.cau_hoi))
        n_chu, n_loi, n_tg = n_chu + len(chu), n_loi + len(loi), n_tg + len(tg)
        co_chu += bool(chu)
        co_loi += bool(loi)
    print(f"mệnh đề: {n_chu} có tín hiệu CHỮ | {n_loi} có tín hiệu LỜI | "
          f"{n_tg} thị giác thuần")
    print(f"câu: {co_chu}/{len(giu)} có ít nhất 1 mệnh đề chữ, "
          f"{co_loi}/{len(giu)} có mệnh đề lời\n")

    anh_nho, ocr_nho = {}, {}

    def anh(c):
        if c.id not in anh_nho:
            anh_nho[c.id] = hop_nhat(
                [k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
        return anh_nho[c.id]

    def moc(c):
        if c.id not in ocr_nho:
            ocr_nho[c.id] = k3.tim(c.cau_hoi, k=a.be)
        return hop_nhat([anh(c), ocr_nho[c.id]], trong_so=[1.0, W3])

    def dinh_tuyen_gop(c):
        """Kênh 3 (gộp) chỉ nhận mệnh đề có tín hiệu chữ/lời."""
        chu, loi, _ = phan_loai(R.tach_truy_van(c.cau_hoi))
        vao = chu + loi
        if not vao:
            return anh(c)                    # không có gì cho kênh 3 -> bỏ nó
        return hop_nhat([anh(c), k3.tim(vao, k=a.be)], trong_so=[1.0, W3])

    def tach_kenh(c):
        """Tách OCR/ASR thành hai kênh, cả hai vẫn nhận CẢ CÂU (đối chứng)."""
        return hop_nhat([anh(c), k_ocr.tim(c.cau_hoi, k=a.be),
                         k_asr.tim(c.cau_hoi, k=a.be)],
                        trong_so=[1.0, W3 / 2, W3 / 2])

    def tach_va_tuyen(c):
        chu, loi, _ = phan_loai(R.tach_truy_van(c.cau_hoi))
        ds, ts = [anh(c)], [1.0]
        if chu:
            ds.append(k_ocr.tim(chu, k=a.be))
            ts.append(W3 / 2)
        if loi:
            ds.append(k_asr.tim(loi, k=a.be))
            ts.append(W3 / 2)
        return hop_nhat(ds, trong_so=ts) if len(ds) > 1 else anh(c)

    def _nho(f):
        c_ = {}

        def g(c):
            if c.id not in c_:
                c_[c.id] = f(c)[:100]
            return c_[c.id]
        return g

    cau_hinh = {
        "1. mốc: run.py": _nho(moc),
        "2. định tuyến, kênh 3 gộp": _nho(dinh_tuyen_gop),
        "3. tách OCR/ASR, KHÔNG tuyến": _nho(tach_kenh),
        "4. tách OCR/ASR + định tuyến": _nho(tach_va_tuyen),
    }
    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
