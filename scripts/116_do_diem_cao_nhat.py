"""
116_do_diem_cao_nhat.py — Điểm CAO NHẤT hệ thống đạt được, và trần còn cách bao xa.

    python scripts/116_do_diem_cao_nhat.py --file dev/tap_de_that.jsonl

BA CON SỐ, ĐỪNG LẪN VỚI NHAU

    1. ĐANG CHẠY   cấu hình mặc định — con số duy nhất được phép hứa
    2. CAO NHẤT    bật hết mọi ứng viên 🟡 — con số LẠC QUAN, chưa cái nào
                   thắng vượt ngưỡng nhiễu, và bật cả cụm thì rủi ro cộng dồn
    3. TRẦN        đáp án có nằm trong bể ứng viên không — giới hạn tuyệt đối
                   của việc XẾP LẠI HẠNG, không phải điểm đạt được

Khoảng cách 1 -> 3 là thứ A92 gọi tên: bài toán còn lại là **xếp hạng**, không
phải tìm kiếm.

CÁC ỨNG VIÊN 🟡 ĐƯỢC GỘP, VÀ VÌ SAO CHỌN ĐÚNG BA CÁI NÀY

| | mục | tác động tới | trạng thái |
| --- | --- | --- | --- |
| `--be-trake 300` | A94 | chỉ TRAKE | +0,0739/+0,0533, 7-3-8 |
| `--trong-so-hoi 0.25` | A93 | chỉ Q&A (11/12 câu) | +0,0154/+0,0154, 4-1-47 |
| văn bản gộp VietOCR | A88/A91 | kênh 3, mọi câu | +0,0144/+0,0000, 4-1-47 |

⚠️ **KHÔNG gộp khuếch tán thời gian τ=2s (A82)** dù nó cũng 🟡 dương. A91 đo
lưới 2×2 và thấy nó **giẫm lên** văn bản gộp: ô "cả hai" thấp hơn ô "chỉ gộp"
ở cả hai mức dung sai. Gộp cả hai là tự chuốc lấy phần trừ đã đo được.

⚠️ Hai cái đầu tác động lên **hai loại câu rời nhau** (TRAKE / Q&A) nên không
thể giẫm nhau. Cái thứ ba chồng lên cả hai — đó là rủi ro thật, và bảng dưới
có dòng tách riêng để đọc được.

⚠️ TRẦN Ở ĐÂY LÀ TRẦN CỦA XẾP HẠNG, KHÔNG PHẢI TRẦN CỦA HỆ THỐNG. Nó giả định
một bộ xếp hạng hoàn hảo trên đúng bể đang có. Câu nào đáp án không lọt vào bể
thì trần của nó là 0 — và A92 đo được 6/49 câu KIS/Q&A rơi vào đó.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan, doc_van_ban_khung        # noqa: E402
from cham_diem import MOC, bao_cao_do_nhay            # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
BE_TRAKE = 300
W_HOI = 0.25
DUNG_SAI = (2.0, 15.0)


def trong_be(uv, c, master, dung_sai) -> bool:
    """Đáp án có nằm trong danh sách ứng viên không (bất kể hạng mấy)?"""
    pts, vid = master.pts_time.values, master.video_id.values
    dung = c.row_id_dung
    if c.loai == "TRAKE":
        dung = [r for b in dung for r in b]
    moc = [(str(vid[r]), float(pts[r])) for r in dung]
    return any(any(v == u.video_id and abs(float(pts[u.row_id]) - t) <= dung_sai
                   for v, t in moc) for u in uv)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    print("dựng kênh 3 trên văn bản GỘP (ocr cũ + VietOCR)…", flush=True)
    k3g = KenhVanBan.tu_bang_khung(
        master, doc_van_ban_khung(a.index), cot="text", ten="gop")

    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    n_loai = pd.Series([c.loai for c in giu]).value_counts().to_dict()
    print(f"\n{len(giu)} câu — {n_loai}\n")

    n1, n3 = {}, {}

    def md(m):
        if m not in n1:
            n1[m] = k1.tim(m, k=a.be)
        return n1[m]

    def van(c, gop):
        k = (c.id, gop)
        if k not in n3:
            n3[k] = (k3g if gop else k3).tim(c.cau_hoi, k=a.be)
        return n3[k]

    def anh(c, w_hoi):
        me = R.tach_truy_van(c.cau_hoi)
        if len(me) == 1:
            return md(me[0])
        if w_hoi == 1.0:
            return hop_nhat([md(m) for m in me])
        ta = [m for m in me if not R.la_menh_de_hoi(m)]
        hoi = [m for m in me if R.la_menh_de_hoi(m)]
        if not ta:
            ta, hoi = me, []
        ds = [md(m) for m in ta] + [md(m) for m in hoi]
        ts = [1.0] * len(ta) + [w_hoi] * len(hoi)
        return hop_nhat(ds, trong_so=ts)

    def dung(w_hoi=1.0, gop=False):
        n = {}

        def g(c):
            if c.id not in n:
                n[c.id] = hop_nhat([anh(c, w_hoi), van(c, gop)],
                                   trong_so=[1.0, W3])[:100]
            return n[c.id]
        return g

    cau_hinh = {
        "1. ĐANG CHẠY (mặc định)": dung(),
        f"2. + mệnh đề hỏi w={W_HOI:g} (A93)": dung(w_hoi=W_HOI),
        "3. + văn bản gộp (A88)": dung(gop=True),
        "4. CAO NHẤT: cả hai": dung(w_hoi=W_HOI, gop=True),
    }
    print(bao_cao_do_nhay(giu, cau_hinh, master))

    # ---------------------------------------------------------------- TRẦN
    print(f"\n{'=' * 70}\nTRẦN — đáp án có LỌT VÀO BỂ không (giả định xếp hạng "
          f"hoàn hảo)\n{'=' * 70}")
    tot = dung(w_hoi=W_HOI, gop=True)
    print(f"  {'dung sai':>10}{'trần':>10}{'đang đạt':>12}{'còn thiếu':>12}")
    print("  " + "-" * 46)
    for ds in DUNG_SAI:
        co = [trong_be(tot(c), c, master, ds) for c in giu]
        tran = float(np.mean(co))
        # điểm đang đạt, cùng dung sai, cấu hình mặc định
        moc_f = dung()
        diem = []
        pts, vid = master.pts_time.values, master.video_id.values
        for c in giu:
            dungs = (c.row_id_dung if c.loai != "TRAKE"
                     else [r for b in c.row_id_dung for r in b])
            m = [(str(vid[r]), float(pts[r])) for r in dungs]
            h = None
            for i, u in enumerate(moc_f(c), 1):
                if any(v == u.video_id and
                       abs(float(pts[u.row_id]) - t) <= ds for v, t in m):
                    h = i
                    break
            diem.append(np.mean([h is not None and h <= k for k in MOC]))
        d = float(np.mean(diem))
        print(f"  {'±' + format(ds, 'g') + 's':>10}{tran:>10.4f}"
              f"{d:>12.4f}{tran - d:>12.4f}")
    print("\n  `trần` = tỷ lệ câu có đáp án ở ĐÂU ĐÓ trong 100 dòng. Nếu xếp"
          "\n  hạng hoàn hảo thì mỗi câu đó được 1,0, nên trần = tỷ lệ đó."
          "\n  `còn thiếu` là phần một bộ XẾP LẠI HẠNG hoàn hảo sẽ lấy được —"
          "\n  không phải phần một kênh truy hồi mới lấy được.")


if __name__ == "__main__":
    main()
