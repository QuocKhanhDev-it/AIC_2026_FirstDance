"""
103_do_dao_cum_idf.py — Đào đáp án Q&A bằng CHẤM ĐIỂM CỤM thay vì đòi chữ HOA.

    python scripts/103_do_dao_cum_idf.py

A84 tìm ra nút thắt thật của Q&A: `TEN`/`TEN_RONG` chỉ khớp cụm bắt đầu bằng
chữ HOA, mà đáp án của kho này phần lớn là **danh từ thường giữa câu**. Nới
regex không cứu được vì điều kiện chữ hoa ở từ ĐẦU vẫn còn.

A88 đổi tình hình: VietOCR đọc ra `'Thịt cá lóc 300g'` ĐÚNG DẤU. Đo trần của
phép trích cụm (có cụm 1–4 từ nào bằng đúng đáp án không):

    CŨ  (ocr + asr)              4/13
    GỘP (ocr + VietOCR + asr)    6/13   <- trần tăng 50%

Nhưng bộ đào hiện tại chỉ lấy được **3/13**. Khoảng cách đó là thứ script này đo.

CÁCH MỚI: sinh MỌI cụm 1–4 từ rồi XẾP HẠNG, không lọc theo hình thức chữ.

    điểm = max(IDF của các token)  −  khoảng cách tới từ khoá / 500

IDF vì đáp án Q&A gần như luôn là **thực thể hiếm**, còn cụm rác là từ phổ
biến. Dùng bảng IDF của CHÍNH kho — `HTV Online` hiếm trong tiếng Việt nhưng
có ở 19.656 khung L26 (A88), chỉ bảng của kho mới hạ được nó.

⚠️ ĐO CẢ TRẦN ở mọi cấu hình. Nếu cách mới chạm trần thì phần còn lại nằm ở
VĂN BẢN, không ở bộ đào — và lúc đó dừng tối ưu bộ đào là đúng.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from bm25 import BM25, doc_van_ban_khung, tach        # noqa: E402
from dap_an import bo_dau, dao_cum, dao_nhieu         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl", GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--k", type=int, default=5, help="số biến thể mỗi khung")
    a = ap.parse_args()

    b = pd.read_parquet(a.index / "ocr_asr.parquet")
    cu = {int(r): f"{o} {s}".strip() for r, o, s in zip(
        b.row_id.values, b.ocr_text.fillna("").values,
        b.asr_text.fillna("").values)}
    gop_b = doc_van_ban_khung(a.index)
    gop = dict(zip(gop_b.row_id.astype(int), gop_b.text.fillna("")))

    print("dựng bảng IDF từ văn bản GỘP…", flush=True)
    idx = BM25([bo_dau(t) for t in gop_b.text.tolist()], bigram=False)
    idf = idx.idf
    print(f"  {len(idf):,} token\n")

    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "QA" and c.dap_an]

    def khop(c, lay, dao_f):
        mong = c.dap_an.strip().lower()
        for r in c.row_id_dung:
            for x in dao_f(lay(r), c.cau_hoi):
                if x.strip().lower() == mong:
                    return True
        return False

    def tran(c, lay, n=4):
        mong = c.dap_an.strip().lower()
        for r in c.row_id_dung:
            tu = [m.group() for m in __import__("re").finditer(
                r"[^\W\d_]+|\d+(?:[.,]\d+)?", lay(r), __import__("re").UNICODE)]
            for k in range(1, n + 1):
                for i in range(len(tu) - k + 1):
                    if " ".join(tu[i:i + k]).lower() == mong:
                        return True
        return False

    cot = {
        "CŨ · regex hoa": (cu, lambda v, q: dao_nhieu(v, q, a.k)),
        "GỘP · regex hoa": (gop, lambda v, q: dao_nhieu(v, q, a.k)),
        "GỘP · cụm+IDF": (gop, lambda v, q: dao_cum(v, q, idf, a.k)),
    }

    print(f"{'câu':<15}{'đáp án':<17}" + "".join(f"{k:>18}" for k in cot)
          + f"{'TRẦN gộp':>11}")
    print("-" * (32 + 18 * len(cot) + 11))
    dem = {k: 0 for k in cot}
    n_tran = 0
    for c in cau:
        o = []
        for k, (lay, f) in cot.items():
            ok = khop(c, lambda r: lay.get(r, ""), f)
            dem[k] += ok
            o.append("✅" if ok else "—")
        t = tran(c, lambda r: gop.get(r, ""))
        n_tran += t
        print(f"{c.id:<15}{c.dap_an[:15]:<17}"
              + "".join(f"{x:>18}" for x in o) + f"{('có' if t else '—'):>11}")
    print()
    for k, n in dem.items():
        print(f"  {k:<20}{n}/{len(cau)}")
    print(f"  {'TRẦN (gộp)':<20}{n_tran}/{len(cau)}   <- mọi phép chọn cụm "
          f"đều chặn ở đây")


if __name__ == "__main__":
    main()
