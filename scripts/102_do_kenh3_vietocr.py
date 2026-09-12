"""
102_do_kenh3_vietocr.py — Kênh 3 trên văn bản GỘP (OCR cũ + VietOCR), và tỷ trọng dấu.

    python scripts/102_do_kenh3_vietocr.py

A88 ghép xong 12 phần VietOCR: 177.321/177.321 khung, tỷ lệ có dấu toàn kho
**7% -> 45%**, riêng khung đáp án **20% -> 82%**. Nhưng văn bản mới **chưa vào
đường chạy nào** — nó nằm ở một file riêng mà `run.py` không đọc.

Đây là lần đầu kênh 3 được cấp THÊM DỮ LIỆU THẬT kể từ khi dựng. Mốc nền hiện
tại: kênh 3 đóng góp **+0,0413, ✅ ổn định** trên nền gopt (A87).

HAI THỨ ĐO CÙNG LÚC

**1. Văn bản gộp có hơn văn bản cũ không.** `bm25.doc_van_ban_khung()` nối
`OCR cũ + VietOCR` — GỘP chứ không thay, vì A76 đo được VietOCR **làm mất một
con số** (`46`) mà OCR cũ đọc được.

⚠️ KHÔNG SUY ĐƯỢC DẤU CỦA HIỆU ỨNG. Nối chuỗi làm tăng TF của từ hai bản cùng
đọc ra (bão hoà theo `k1`, tức như một phép tăng trọng số nhẹ) NHƯNG cũng làm
tăng `dl`, mà BM25 **phạt độ dài** qua `b`. Hai hiệu ứng ngược chiều. Độ dài
trung vị đi từ 489 lên 510 ký tự (+4%) — đủ để `b` cắn.

**2. Tỷ trọng nhánh CÓ DẤU (`alpha`).** `α·có_dấu + (1−α)·không_dấu`; α=0,5 là
hành vi cũ y hệt. Lập luận: khi văn bản có thêm 45% chữ có dấu thì nhánh có dấu
sắc hơn, vì IDF của từ có dấu cao hơn từ đã bỏ dấu. Nhưng nhánh không dấu tồn
tại để **cứu truy vấn gõ thiếu dấu và OCR đọc sai dấu** — hạ nó xuống có thể
mất nhiều hơn được. Dò 0,5 / 0,6 / 0,7 / 0,8.

⚠️ α chỉ đổi lúc CHẤM, không đổi chỉ mục — nên bốn mức dùng chung một chỉ mục,
không dựng lại bốn lần. Chỉ mục cũ được thả trước khi dựng chỉ mục gộp (máy
7,7 GB).
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan, doc_van_ban_khung        # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
CAC_ALPHA = (0.5, 0.6, 0.7, 0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)} câu\n")

    anh = {c.id: hop_nhat([k1.tim(m, k=a.be)
                           for m in R.tach_truy_van(c.cau_hoi)]) for c in giu}
    del k1
    gc.collect()

    ocr = {}

    def quet(ten, bang, alphas):
        """Dựng chỉ mục, tính ứng viên cho từng α, rồi THẢ chỉ mục."""
        k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten=ten)
        print(f"  {ten}: {len(k3):,} khung có chữ", flush=True)
        for al in alphas:
            k3.alpha = al
            ocr[(ten, al)] = {c.id: k3.tim(c.cau_hoi, k=a.be) for c in giu}
        del k3
        gc.collect()

    quet("cũ", pd.read_parquet(a.index / "ocr_asr.parquet"), [0.5])
    quet("gộp", doc_van_ban_khung(a.index), CAC_ALPHA)
    print()

    def hop(khoa):
        return lambda c: hop_nhat([anh[c.id], ocr[khoa][c.id]],
                                  trong_so=[1.0, W3])[:100]

    cau_hinh = {"MỐC: OCR cũ (α=0,5)": hop(("cũ", 0.5))}
    for al in CAC_ALPHA:
        cau_hinh[f"GỘP + VietOCR, α={al:g}"] = hop(("gộp", al))
    cau_hinh["chỉ kênh 1 (chẩn đoán)"] = lambda c: anh[c.id][:100]

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
