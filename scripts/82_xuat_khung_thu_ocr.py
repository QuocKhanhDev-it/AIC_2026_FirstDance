"""
82_xuat_khung_thu_ocr.py — Xuất danh sách khung để máy có GPU chạy thử VietOCR.

    python scripts/82_xuat_khung_thu_ocr.py

CÂU HỎI CẦN TRẢ LỜI, VÀ VÌ SAO NÓ RẺ

A68 đo được hai rào cản của việc đào đáp án Q&A từ văn bản:

    ocr_text có dấu tiếng Việt : 31% (944/3.000 mẫu)
    asr_text có dấu            : 100%, nhưng viết SỐ bằng CHỮ

Rào cản 1 chữa được nếu OCR đọc ra chữ CÓ DẤU. `vietnamese-news-video-ocr` của
một nhóm khác dùng PaddleOCR (dò chữ) + **VietOCR** (đọc chữ) — VietOCR là model
chuyên tiếng Việt nên trả về chữ có dấu.

Nhưng chạy lại OCR cho 177.321 ảnh là hàng chục giờ GPU. Trước khi bỏ ra chừng
đó, chỉ cần trả lời MỘT câu:

> Đáp án vàng của 13 câu Q&A có xuất hiện **ĐÚNG DẤU** trong OCR mới không?

Hiện tại là 7/13 và **toàn khớp nhờ bỏ dấu**. Câu đó đo được trên đúng **63
khung đáp án** — vài giây GPU, không phải hàng chục giờ.

File này xuất hai nhóm khung:

  * `qa`   — 63 khung đáp án của 13 câu Q&A: trả lời câu hỏi trên
  * `toc`  — 200 khung ngẫu nhiên: đo s/ảnh để ước chi phí cả kho

⚠️ KHÔNG XUẤT ĐÁP ÁN. File đi sang Kaggle và có thể lọt vào log công khai; so
đáp án làm ở máy này.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl", GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--so-toc-do", type=int, default=200)
    ap.add_argument("--ra", default=GOC / "dev" / "khung_thu_ocr.jsonl", type=Path)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "QA" and c.dap_an]

    rid = sorted({r for c in cau for r in c.row_id_dung})
    rng = np.random.default_rng(0)
    them = rng.choice(len(master), size=a.so_toc_do, replace=False)

    with a.ra.open("w", encoding="utf-8", newline="\n") as f:
        for nhom, ids in (("qa", rid), ("toc", them.tolist())):
            for r in ids:
                d = master.iloc[int(r)]
                kf = d.kf_name if pd.notna(d.kf_name) else f"{int(d.kf_n):03d}.jpg"
                f.write(json.dumps({"nhom": nhom, "row_id": int(r),
                                    "video_id": str(d.video_id),
                                    "kf_name": str(kf)},
                                   ensure_ascii=False) + "\n")

    print(f"✅ {a.ra}")
    print(f"   {len(rid)} khung đáp án của {len(cau)} câu Q&A")
    print(f"   {a.so_toc_do} khung ngẫu nhiên để đo tốc độ")
    print(f"   KHÔNG chứa đáp án — an toàn để đưa lên Kaggle")


if __name__ == "__main__":
    main()
