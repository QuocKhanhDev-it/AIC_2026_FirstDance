"""
80_do_gom_doan_ocr.py — Gom khung liên tiếp cùng OCR thành ĐOẠN: có bao nhiêu đoạn?

    python scripts/80_do_gom_doan_ocr.py

Ý TƯỞNG ĐEM ĐO KHẢ THI (học từ `segment_topics.py` của một nhóm khác)

OCR là dòng ticker rời rạc từng khung ("06:30:11 | TIN CHÍNH | …"). Câu hỏi thì
diễn giải lại nội dung. BM25 khớp mặt chữ nên không bắt được.

Cách họ làm: gom các khung LIÊN TIẾP cùng video có OCR giống nhau (Jaccard trên
token) thành một đoạn, rồi **một lượt LLM cho mỗi đoạn** tóm tắt nội dung +
trích thực thể neo. Gán câu tóm tắt đó cho MỌI khung trong đoạn.

Vì sao đáng đo trước khi làm: nó biến 177.321 lượt gọi LLM (bất khả thi) thành
"số đoạn" lượt. **Nhưng số đoạn là bao nhiêu thì chưa ai biết** — nếu OCR đổi
liên tục thì số đoạn xấp xỉ số khung và ý tưởng sụp ngay ở khâu chi phí.

Script này CHỈ đếm. Không sinh tóm tắt, không gọi LLM. Trả lời đúng một câu:
**bao nhiêu lượt gọi LLM, và mỗi đoạn dài bao nhiêu?**
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))


def bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--nguong", type=float, nargs="*",
                    default=[0.3, 0.5, 0.7],
                    help="ngưỡng Jaccard để coi hai khung liền kề là CÙNG đoạn")
    ap.add_argument("--cot", default="ocr_text", help="ocr_text | asr_text | text")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    d = master[["row_id", "video_id"]].merge(
        bang[["row_id", a.cot]], on="row_id", how="left")
    d[a.cot] = d[a.cot].fillna("")
    tok = [set(bo_dau(x).split()) for x in d[a.cot]]
    vid = d.video_id.values
    n = len(d)
    co_chu = sum(1 for x in tok if x)
    print(f"cột `{a.cot}`: {co_chu:,}/{n:,} khung có chữ "
          f"({co_chu / n * 100:.1f}%)")
    dai = [len(x) for x in tok if x]
    print(f"  token/khung: trung vị {int(np.median(dai))}, "
          f"p90 {int(np.quantile(dai, .9))}\n")

    print(f"{'ngưỡng':>8}{'số đoạn':>11}{'khung/đoạn':>13}"
          f"{'đoạn 1 khung':>15}{'giờ LLM @1s':>13}")
    print("-" * 62)
    for ng in a.nguong:
        dau_doan = np.ones(n, dtype=bool)
        for i in range(1, n):
            if vid[i] == vid[i - 1] and jaccard(tok[i], tok[i - 1]) >= ng:
                dau_doan[i] = False
        so_doan = int(dau_doan.sum())
        # độ dài mỗi đoạn
        moc = np.flatnonzero(dau_doan)
        do_dai = np.diff(np.append(moc, n))
        mot = int((do_dai == 1).sum())
        print(f"{ng:>8.1f}{so_doan:>11,}{do_dai.mean():>13.1f}"
              f"{mot:>10,} ({mot / so_doan * 100:>3.0f}%){so_doan / 3600:>13.1f}")

    print("\n⚠️ Cột `giờ LLM` giả định 1 giây/đoạn — lạc quan cho API thật.\n"
          "   Đoạn CHỈ MỘT KHUNG là đoạn không gom được gì: tóm tắt nó tốn đúng\n"
          "   một lượt gọi mà chẳng lan tín hiệu sang khung nào.")


if __name__ == "__main__":
    main()
