"""
90_do_so_bang_chu.py — ASR viết SỐ bằng CHỮ: chuyển lại có cứu được câu nào không?

    python scripts/90_do_so_bang_chu.py

A68 đo được hai rào cản của việc đào đáp án Q&A từ văn bản. A76 vừa vá rào cản
thứ nhất (OCR không dấu) bằng VietOCR + gộp. Đây là rào cản thứ hai:

    asr_text : 100% có dấu, nhưng viết SỐ bằng CHỮ

Và **7/13 đáp án Q&A là số** (`46`, `2,15`, `7`, `20`, `2`, `200g`, `1204`).
Nếu ASR đọc "bốn mươi sáu" thì chuỗi `46` không bao giờ khớp.

VÌ SAO ĐO CHIỀU NGƯỢC (số -> chữ), KHÔNG PHẢI CHIỀU XUÔI

Thư viện `vietnam-number` làm chiều **chữ -> số**, và đó là thứ cần cho bản
chạy thật. Nhưng để biết có ĐÁNG cài hay không thì chiều ngược trả lời rẻ hơn
và chắc hơn: sinh cách đọc tiếng Việt của đáp án rồi tìm trong ASR.

  * viết số thành chữ dễ hơn và không mơ hồ;
  * nó trả lời đúng câu cần hỏi — "dạng chữ của đáp án CÓ nằm trong ASR không";
  * và nếu câu trả lời là KHÔNG thì không thư viện nào cứu được, khỏi cài.

Sinh nhiều biến thể cho mỗi số (`hai mươi tư`/`hai mươi bốn`, `linh`/`lẻ`,
`nghìn`/`ngàn`) để không đếm thiếu — phép đo này chỉ có ý nghĩa nếu nó rộng
lượng với hướng đang xét.

⚠️ Script này KHÔNG cần kênh nào, không nạp ma trận. Chỉ đọc `ocr_asr.parquet`.
"""

import argparse
import itertools
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402

DON = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def doc_hai_chu_so(n: int) -> list[str]:
    """10–99 -> các cách đọc. Tiếng Việt có biến âm ở 1, 4, 5 sau hàng chục."""
    chuc, dv = divmod(n, 10)
    dau = "mười" if chuc == 1 else f"{DON[chuc]} mươi"
    if dv == 0:
        return [dau]
    duoi = {1: ["mốt"] if chuc > 1 else ["một"],
            4: ["tư", "bốn"] if chuc > 1 else ["bốn"],
            5: ["lăm", "nhăm"]}.get(dv, [DON[dv]])
    return [f"{dau} {d}" for d in duoi]


def doc_so(n: int) -> list[str]:
    """Số nguyên 0–9999 -> danh sách cách đọc tiếng Việt (nhiều biến thể)."""
    if n < 10:
        return [DON[n]]
    if n < 100:
        return doc_hai_chu_so(n)
    if n < 1000:
        tram, du = divmod(n, 100)
        dau = f"{DON[tram]} trăm"
        if du == 0:
            return [dau]
        if du < 10:
            return [f"{dau} {le} {DON[du]}" for le in ("linh", "lẻ")]
        return [f"{dau} {x}" for x in doc_hai_chu_so(du)]
    nghin, du = divmod(n, 1000)
    ra = []
    for tn in ("nghìn", "ngàn"):
        dau = f"{DON[nghin]} {tn}"
        if du == 0:
            ra.append(dau)
        elif du < 10:
            ra += [f"{dau} {le} {DON[du]}" for le in ("linh", "lẻ")]
        elif du < 100:
            ra += [f"{dau} không trăm {x}" for x in doc_hai_chu_so(du)]
            ra += [f"{dau} {x}" for x in doc_hai_chu_so(du)]
        else:
            ra += [f"{dau} {x}" for x in doc_so(du)]
    return ra


def cach_doc(dap_an: str) -> list[str]:
    """Mọi cách đọc của phần SỐ trong đáp án. [] nếu đáp án không có số."""
    so = re.findall(r"\d+(?:[.,]\d+)?", dap_an)
    if not so:
        return []
    phan = []
    for s in so:
        if "," in s or "." in s:
            a, b = re.split(r"[.,]", s, maxsplit=1)
            # "2,15" đọc là "hai phẩy mười lăm" hoặc "hai phẩy một năm"
            roi = " ".join(DON[int(c)] for c in b)
            phan.append([f"{x} phẩy {y}"
                         for x in doc_so(int(a))
                         for y in set(doc_so(int(b))) | {roi}])
        else:
            n = int(s)
            if n > 9999:
                phan.append([s])
            else:
                # số dài cũng hay được đọc rời từng chữ số
                roi = " ".join(DON[int(c)] for c in s)
                phan.append(list(set(doc_so(n)) | {roi}))
    return [" ".join(x) for x in itertools.product(*phan)]


def bo_dau(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl", GOC / "dev" / "tap_de_thi_thu.jsonl"])
    a = ap.parse_args()

    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    asr = dict(zip(bang.row_id.astype(int), bang.asr_text.fillna("")))
    ocr = dict(zip(bang.row_id.astype(int), bang.ocr_text.fillna("")))

    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "QA" and c.dap_an]

    print(f"{len(cau)} câu Q&A\n")
    print(f"{'câu':<16}{'đáp án':<10}{'số?':>5}{'chuỗi số trong ASR':>20}"
          f"{'DẠNG CHỮ trong ASR':>21}")
    print("-" * 74)

    n_so = so_thang = chu_thang = 0
    for c in cau:
        vang = c.dap_an.strip()
        doc = cach_doc(vang)
        if not doc:
            print(f"{c.id:<16}{vang[:9]:<10}{'·':>5}{'—':>20}{'—':>21}")
            continue
        n_so += 1
        van = bo_dau(" ".join(str(asr.get(r, "")) for r in c.row_id_dung))
        # chuỗi số nguyên văn có sẵn trong ASR chưa?
        co_so = any(bo_dau(s) in van
                    for s in re.findall(r"\d+(?:[.,]\d+)?", vang))
        khop = [d for d in doc if bo_dau(d) in van]
        so_thang += co_so
        chu_thang += bool(khop)
        print(f"{c.id:<16}{vang[:9]:<10}{'CÓ':>5}"
              f"{('có' if co_so else '—'):>20}"
              f"{(khop[0][:20] if khop else '—'):>21}")

    print(f"\n{n_so}/{len(cau)} đáp án có chứa số")
    print(f"  {so_thang}/{n_so} đã khớp bằng CHỮ SỐ trong ASR (không cần chuyển)")
    print(f"  {chu_thang}/{n_so} tìm thấy DẠNG CHỮ trong ASR "
          f"<- phần chuyển đổi có thể cứu")
    if chu_thang == 0:
        print("\n=> ASR KHÔNG đọc số của các đáp án này ra chữ. Cài thư viện")
        print("   chuyển chữ->số sẽ không cứu được câu nào. ĐÓNG hướng này.")


if __name__ == "__main__":
    main()
