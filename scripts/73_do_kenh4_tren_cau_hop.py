"""
73_do_kenh4_tren_cau_hop.py — Kênh 4 có ích trên đúng loại câu nó HỢP không?

    python scripts/73_do_kenh4_tren_cau_hop.py

A61 đo kênh 4 trên toàn bộ 52 câu và ra 0,0125 khi đứng một mình. Nhưng trung
bình trên mọi câu che mất một khả năng: kênh 4 chỉ hợp với câu **có nói tới vật
thể**, và nếu chỉ 1/4 số câu như thế thì tín hiệu bị pha loãng bốn lần.

Soi được: **26/52 câu đề thật nhắc tới một nhãn ĐẶC TRƯNG (IDF≥3) có mặt trên
khung đáp án** — ví dụ "bảng trắng" (Whiteboard), "xe máy" (Motorcycle), "ô"
(Umbrella). Trên đúng 26 câu đó, kênh 4 có mọi điều kiện để thắng.

⚠️ ĐÂY LÀ TRẦN TRÊN, KHÔNG PHẢI MỘT CẤU HÌNH DÙNG ĐƯỢC

Cách chọn 26 câu này **dùng tới đáp án** (phải biết khung đúng mới biết nó mang
nhãn gì). Ngày thi không có thông tin đó. Nên con số ở đây trả lời câu hỏi
"kênh 4 giỏi nhất có thể tới đâu", chứ không phải "bật kênh 4 thì được bao
nhiêu".

Ý nghĩa: nếu ngay cả ở trần trên nó vẫn ~0 thì hướng objects **chết hẳn**, khỏi
bàn tiếp chuyện định tuyến hay bảng nhãn. Nếu nó khá thì mới đáng nghĩ cách
nhận diện "câu hợp với objects" mà KHÔNG cần đáp án.
"""

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from objects import KenhObjects, nap_bang_nhan        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
IDF_DAC_TRUNG = 3.0


def bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    obj = pd.read_parquet(a.index / "objects.parquet")
    cau = tap_dev.doc(a.file)

    n_kf = obj.row_id.nunique()
    idf = (n_kf / obj.groupby("label").row_id.nunique()).apply(math.log)
    theo_row = obj.groupby("row_id").label.apply(list)
    tu_kd = {r.nhan_en: [bo_dau(x) for x in r.tu_kd]
             for r in nap_bang_nhan().itertuples()}

    def hop(c) -> bool:
        rid = ([x for b in c.row_id_dung for x in b]
               if c.loai == "TRAKE" else c.row_id_dung)
        nhan = {x for i in rid for x in theo_row.get(i, [])}
        q = bo_dau(c.cau_hoi)
        return any(idf.get(n, 0) >= IDF_DAC_TRUNG and n in tu_kd
                   and any(re.search(rf"\b{re.escape(t)}\b", q)
                           for t in tu_kd[n])
                   for n in nhan)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    k4 = KenhObjects(str(a.index), master)

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    chon = [c for c in giu if hop(c)]
    print(f"{a.file.name}: {len(chon)}/{len(giu)} câu HỢP với objects "
          f"(câu nhắc tới nhãn IDF≥{IDF_DAC_TRUNG:g} có trên khung đáp án)\n")

    nen = {}

    def phan(c):
        if c.id not in nen:
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
            nen[c.id] = (anh, k3.tim(c.cau_hoi, k=a.be), k4.tim(c.cau_hoi, k=a.be))
        return nen[c.id]

    def _nho(f):
        n = {}

        def g(c):
            if c.id not in n:
                n[c.id] = f(c)[:100]
            return n[c.id]
        return g

    cau_hinh = {
        "1. mốc: run.py": _nho(lambda c: hop_nhat(list(phan(c)[:2]),
                                                  trong_so=[1.0, W3])),
        "2. + kênh 4 (0,25)": _nho(lambda c: hop_nhat(
            list(phan(c)), trong_so=[1.0, W3, 0.25])),
        "3. + kênh 4 (0,5)": _nho(lambda c: hop_nhat(
            list(phan(c)), trong_so=[1.0, W3, 0.5])),
        "4. chỉ kênh 4 (chẩn đoán)": _nho(lambda c: phan(c)[2]),
        "5. chỉ kênh 1 (chẩn đoán)": _nho(lambda c: phan(c)[0]),
    }
    print(bao_cao_do_nhay(chon, cau_hinh, master))
    print("\n⚠️ TRẦN TRÊN: cách chọn 26 câu này dùng tới đáp án, ngày thi không "
          "có. Đọc là 'kênh 4 giỏi nhất tới đâu', không phải 'bật thì được bao "
          "nhiêu'.")


if __name__ == "__main__":
    main()
