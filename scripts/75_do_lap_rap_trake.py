"""
75_do_lap_rap_trake.py — Khâu LẮP RÁP TRAKE mất bao nhiêu điểm?

    python scripts/75_do_lap_rap_trake.py

TẦNG CHƯA AI ĐO — chính docstring `cham_diem.diem_trake_bai_nop()` đã ghi thế.

`cham_diem.cham()` chấm TRAKE bằng `diem_trake()`: với mỗi sự kiện, tìm xem
khung đúng có nằm trong danh sách ứng viên không. Đó là chấm **KÊNH** — trả lời
*"kênh có tìm ra các sự kiện không"*.

BTC thì chấm **BÀI NỘP**: mỗi dòng là một BỘ N khung, và **vị trí i chỉ được so
với sự kiện i**. Kênh tìm ra đủ ba sự kiện nhưng `dung_trake()` lắp sai vị trí
thì `diem_trake()` cho điểm cao còn BTC cho **0**.

Nghĩa là mọi con số TRAKE trong repo — kể cả các dòng TRAKE trong A54, A59,
A60 — đều đang đo tầng KÊNH, không phải tầng NỘP. Script này đo cả hai để biết
khoảng cách.

Đo trên 3 câu TRAKE của đề thật + 14 câu `tap_dev_trake.jsonl`. 14 câu kia là
câu tự soạn nên KHÔNG dùng để so kênh (A50/A58), nhưng ở đây ta không so kênh —
ta đo **cơ chế lắp ráp**, thứ mà câu tự soạn kiểm được: thứ tự sự kiện và số
Frame ID là ràng buộc hình thức, không phụ thuộc câu hỏi hay dễ.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import (DUNG_SAI_CHINH, DUNG_SAI_KIEM,  # noqa: E402
                       diem_trake, diem_trake_bai_nop, no_cua_so, _hang)
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
MOC = (1, 5, 20, 50, 100)


def diem_trake_bai_nop_da(cac_dong, dung_moi_su_kien, gioi_han: int = 100):
    """Như `cham_diem.diem_trake_bai_nop` nhưng mỗi vị trí là một TẬP row_id.

    Cần vì bài nộp mang `(video_id, frame_idx)`, mà A5.7: một cặp như thế có
    thể ứng với nhiều `row_id`. Khớp bất kỳ cái nào cũng là đúng — nếu chỉ so
    một row_id đại diện thì ta tự chấm mình sai ở 614 keyframe.
    """
    n = len(dung_moi_su_kien)
    if not n:
        return 0.0
    r = []
    for dong in cac_dong[:gioi_han]:
        khop = sum(1 for i, tap in enumerate(dong[:n])
                   if tap & dung_moi_su_kien[i])
        r.append(khop / n)
    if not r:
        return 0.0
    return sum(max(r[:k], default=0.0) for k in MOC) / len(MOC)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl", GOC / "dev" / "tap_dev_trake.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "TRAKE"]
    giu = [c for c in cau
           if not any(k1.co_du(R.tach_truy_van(m))
                      for m in R.tach_su_kien(c.cau_hoi))]
    print(f"{len(giu)}/{len(cau)} câu TRAKE đo được "
          f"(bỏ câu thiếu chuỗi trong cache)\n")

    # (video_id, frame_idx) -> tập row_id. Xem A5.7.
    tra = {}
    for r_, v_, f_ in zip(master.row_id.values, master.video_id.values,
                          master.frame_idx.values):
        tra.setdefault((v_, int(f_)), set()).add(int(r_))

    for ds_giay in (DUNG_SAI_CHINH, DUNG_SAI_KIEM):
        kenh_tb = nop_tb = 0.0
        bang = []
        for c in giu:
            su_kien = R.tach_su_kien(c.cau_hoi)
            cac = []
            for sk in su_kien:
                anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(sk)])
                cac.append(hop_nhat([anh, k3.tim(sk, k=a.be)],
                                    trong_so=[1.0, W3]))
            dung = [no_cua_so(b, master, ds_giay) for b in c.row_id_dung]

            # tầng KÊNH — cách cham() đang chấm
            d_kenh = diem_trake([_hang(cac[i], dung[i]) if i < len(cac) else None
                                 for i in range(len(dung))])
            # tầng NỘP — cách BTC chấm. `dung_trake` trả (video_id, frame_idxs);
            # phải quy về row_id để so. ⚠️ A5.7: 614 keyframe dùng chung
            # frame_idx trong cùng video, nên một (video, frame) có thể ứng
            # với NHIỀU row_id — khớp bất kỳ cái nào cũng là đúng.
            dong = R.dung_trake(cac, master)
            dong_rid = [[tra.get((d.video_id, f), set()) for f in d.frame_idxs]
                        for d in dong]
            d_nop = diem_trake_bai_nop_da(dong_rid, dung)
            kenh_tb += d_kenh
            nop_tb += d_nop
            bang.append((c.id, len(dung), len(dong), d_kenh, d_nop))

        n = len(giu)
        print(f"── dung sai ±{ds_giay:g}s " + "─" * 46)
        print(f"{'câu':<18}{'sk':>3}{'dòng':>6}{'KÊNH':>9}{'NỘP':>9}{'mất':>9}")
        for i, nsk, ndong, dk, dn in bang:
            print(f"{i:<18}{nsk:>3}{ndong:>6}{dk:>9.4f}{dn:>9.4f}"
                  f"{dn - dk:>+9.4f}")
        print(f"{'TRUNG BÌNH':<18}{'':>3}{'':>6}{kenh_tb/n:>9.4f}"
              f"{nop_tb/n:>9.4f}{(nop_tb - kenh_tb)/n:>+9.4f}\n")


if __name__ == "__main__":
    main()
