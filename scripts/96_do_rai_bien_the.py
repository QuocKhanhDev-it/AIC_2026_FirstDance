"""
96_do_rai_bien_the.py — Q&A: rải NHIỀU biến thể `answer` qua nhiều dòng.

    python scripts/96_do_rai_bien_the.py

BTC cho **100 dòng** mỗi gói và **không phạt dòng sai** (PHẦN C). Nhưng
`dap_an.dao()` chỉ trả về MỘT chuỗi cho mỗi khung, nên nếu chuỗi đó sai thì cả
100 dòng cùng sai — kể cả khi ứng viên thứ hai đúng.

Ý: với `n_dong` dòng đầu, phát ra `k` dòng cho mỗi khung, mỗi dòng một biến thể
`answer` khác nhau. Chi phí là chỗ: `k` biến thể của một khung chiếm chỗ của
`k−1` khung khác.

⚠️ ĐÁNH ĐỔI CÓ THẬT, và nó không suy được. Điểm là trung bình R@{1,5,20,50,100}:

  * thêm biến thể ở hạng 2 -> nếu biến thể đúng thì R@5, R@20… được cứu, nhưng
    R@1 thì không (hạng 1 vẫn là biến thể cũ);
  * mà nó đẩy khung xếp thứ hai xuống hạng 3 — nếu KHUNG mới là thứ đúng thì
    ta vừa tự làm hại mình.

Đo trên **13 câu Q&A có đáp án vàng**. n nhỏ, nên đọc thắng–thua–hoà chứ đừng
đọc mỗi hiệu trung bình.

DÒNG `trần` LÀ CHẨN ĐOÁN QUAN TRỌNG NHẤT: chấm như thể MỌI biến thể đều được
thử ở đúng khung đó. Nó cho biết việc rải biến thể có TRẦN bao nhiêu — nếu
trần cũng thấp thì vấn đề nằm ở khâu ĐÀO, không phải khâu rải.
"""

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dap_an import dao, dao_nhieu                     # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5


def gan_mot(uv, cau_hoi, van, mac_dinh="không rõ"):
    """Hành vi HIỆN TẠI: mỗi dòng một `answer`, đào từ chính khung đó."""
    ra = []
    for c in uv:
        x = copy.copy(c)
        x.meta = {**c.meta, "answer": dao(van.get(c.row_id, ""), cau_hoi)
                  or mac_dinh}
        ra.append(x)
    return ra


def gan_rai(uv, cau_hoi, van, n_dong, k, mac_dinh="không rõ", gioi_han=100):
    """`n_dong` khung đầu được phát ra tối đa `k` dòng, mỗi dòng một biến thể."""
    ra = []
    for i, c in enumerate(uv):
        bien = (dao_nhieu(van.get(c.row_id, ""), cau_hoi, k)
                if i < n_dong else
                [dao(van.get(c.row_id, ""), cau_hoi)])
        for b in (bien or [None]):
            x = copy.copy(c)
            x.meta = {**c.meta, "answer": b or mac_dinh}
            ra.append(x)
            if len(ra) >= gioi_han:
                return ra
    return ra


def gan_tran(uv, cau_hoi, van, vang, k=5, mac_dinh="không rõ"):
    """CHẨN ĐOÁN: dòng đúng nếu BẤT KỲ biến thể nào của khung đó đúng."""
    mong = vang.strip().lower()
    ra = []
    for c in uv:
        bien = dao_nhieu(van.get(c.row_id, ""), cau_hoi, k)
        trung = next((b for b in bien if b.strip().lower() == mong), None)
        x = copy.copy(c)
        x.meta = {**c.meta, "answer": trung or (bien[0] if bien else mac_dinh)}
        ra.append(x)
    return ra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    van = {int(r): f"{o} {s}".strip() for r, o, s in zip(
        bang.row_id.values, bang.ocr_text.fillna("").values,
        bang.asr_text.fillna("").values)}

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "QA" and c.dap_an]
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)} câu Q&A có đáp án vàng\n")

    tho = {}

    def nen(c):
        if c.id not in tho:
            anh = hop_nhat([k1.tim(m, k=a.be)
                            for m in R.tach_truy_van(c.cau_hoi)])
            tho[c.id] = hop_nhat([anh, k3.tim(c.cau_hoi, k=a.be)],
                                 trong_so=[1.0, W3])
        return tho[c.id]

    def nho(f):
        c_ = {}

        def g(c):
            if c.id not in c_:
                c_[c.id] = f(c)[:100]
            return c_[c.id]
        return g

    cau_hinh = {
        "MỐC: 1 biến thể/dòng": nho(
            lambda c: gan_mot(nen(c), c.cau_hoi, van)),
    }
    for n_dong, k in ((10, 2), (10, 3), (30, 2), (30, 3)):
        cau_hinh[f"rải {k} biến thể, {n_dong} khung đầu"] = nho(
            (lambda n, k: lambda c: gan_rai(nen(c), c.cau_hoi, van, n, k))(
                n_dong, k))
    cau_hinh["TRẦN: mọi biến thể (chẩn đoán)"] = nho(
        lambda c: gan_tran(nen(c), c.cau_hoi, van, c.dap_an))

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
