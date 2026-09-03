"""
91_do_ngan_sach_trake.py — TRAKE: chia 100 dòng cho mấy video, theo tỉ lệ nào?

    python scripts/91_do_ngan_sach_trake.py

A89 vừa loại trừ một nghi phạm và chỉ ra nghi phạm thật.

**Loại trừ:** bốn cách chấm điểm video (tổng / tổng-log / điều hoà / min) chọn
ra gần như CÙNG một video — 10/17 đúng ở hạng 1, **15/17 trong top-5**, ba cách
đầu giống hệt nhau tới từng câu. Cách chấm là nút trơ.

**Nghi phạm thật:** `lap_kbest` lấy top-5 video rồi chia 100 dòng theo tỉ lệ
**50/25/15/7/3**. Oracle thì dồn **cả 100 dòng** vào đúng video chứa đáp án và
được 0,4576 so với 0,3547 — chênh 0,1029.

Nhưng 15/17 video đúng ĐÃ nằm trong top-5. Nghĩa là phần lớn khoảng cách tới
oracle không phải "chọn trượt video", mà là **video đúng chỉ được cấp một phần
ngân sách**: nằm ở hạng 2 thì chỉ có 25 dòng thay vì 100, nên K-best chỉ sinh
được 25 giả thuyết thay vì 100.

ĐÁNH ĐỔI, và đó là thứ phải đo chứ không suy được

Dồn hết cho hạng 1 thì 10/17 câu được đủ 100 dòng — nhưng **5/17 câu có video
đúng ở hạng 2–5 mất trắng**. Trải đều thì ngược lại. Điểm tối ưu nằm ở đâu là
câu hỏi thực nghiệm; phân bố hạng của video đúng (10 ở hạng 1, thêm 5 ở hạng
2–5) không tự nói ra đáp số vì mỗi dòng thêm cho một video có lợi ích giảm dần.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import no_cua_so                       # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

_s = importlib.util.spec_from_file_location(
    "m78", GOC / "scripts" / "78_do_kbest_trake.py")
m78 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m78)

_s9 = importlib.util.spec_from_file_location(
    "m89", GOC / "scripts" / "89_do_chon_video_trake.py")
m89 = importlib.util.module_from_spec(_s9)
_s9.loader.exec_module(m89)

W3 = 0.5
DUNG_SAI = (2.0, 15.0)

# Các cách chia 100 dòng cho top-N video, từ dồn nhất tới trải nhất.
NGAN_SACH = {
    "1. dồn hết hạng 1        (100)": [1.0],
    "2. 70/30                       ": [0.7, 0.3],
    "3. 50/30/20                    ": [0.5, 0.3, 0.2],
    "4. 50/25/15/7/3   ← MỐC        ": [0.5, 0.25, 0.15, 0.07, 0.03],
    "5. 40/25/15/12/8               ": [0.4, 0.25, 0.15, 0.12, 0.08],
    "6. trải đều 5 (20 mỗi video)   ": [0.2] * 5,
    "7. trải đều 10 (10 mỗi video)  ": [0.1] * 10,
}


def lap(cac, master, ty_le, so_dong=100):
    """Y hệt `m89.lap` nhưng ngân sách dòng chia theo `ty_le`."""
    pts = master.pts_time.values
    diem_v, theo_video = m89.xep_video(cac, m89.nhan)
    if not diem_v:
        return []
    xep = sorted(diem_v, key=lambda v: -diem_v[v])[:len(ty_le)]
    ngan = [max(1, round(so_dong * w)) for w in ty_le]
    ra = []
    for v, k in zip(xep, ngan):
        uv = theo_video[v]
        for x in uv:
            x.sort(key=lambda t: -t[1])
        ra += m78.beam_video([x[:20] for x in uv], pts, k)
        if len(ra) >= so_dong:
            break
    return ra[:so_dong]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl",
        GOC / "dev" / "tap_dev_trake.jsonl"])
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
    print(f"{len(giu)}/{len(cau)} câu TRAKE đo được\n")

    nen, v_dung = {}, {}
    for c in giu:
        cac = []
        for sk in R.tach_su_kien(c.cau_hoi):
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(sk)])
            cac.append(hop_nhat([anh, k3.tim(sk, k=a.be)], trong_so=[1.0, W3]))
        nen[c.id] = cac
        v_dung[c.id] = master.video_id.iloc[c.row_id_dung[0][0]]

    # Hạng của video đúng — để đọc bảng dưới có căn cứ.
    hang = []
    for c in giu:
        diem_v, _ = m89.xep_video(nen[c.id], m89.nhan)
        xep = sorted(diem_v, key=lambda v: -diem_v[v])
        hang.append(xep.index(v_dung[c.id]) + 1
                    if v_dung[c.id] in diem_v else 999)
    print("Hạng của video ĐÚNG: " +
          "  ".join(f"h{h}:{hang.count(h)}" for h in sorted(set(hang))))
    print()

    for ds in DUNG_SAI:
        print(f"── dung sai ±{ds:g}s " + "─" * 34)
        print(f"{'ngân sách':<33}{'điểm':>9}{'hiệu':>9}")
        moc = None
        for ten, tl in NGAN_SACH.items():
            tong = 0.0
            for c in giu:
                dung = [no_cua_so(b, master, ds) for b in c.row_id_dung]
                tong += m78.diem_bai_nop(
                    [[{r} for r in d] for d in lap(nen[c.id], master, tl)],
                    dung)
            d = tong / len(giu)
            if "MỐC" in ten:
                moc = d
            print(f"{ten:<33}{d:>9.4f}" +
                  ("" if moc is None or "MỐC" in ten else f"{d - moc:>+9.4f}"))
        # oracle: cả 100 dòng vào ĐÚNG video chứa đáp án
        tong = 0.0
        for c in giu:
            dung = [no_cua_so(b, master, ds) for b in c.row_id_dung]
            tong += m78.diem_bai_nop(
                [[{r} for r in d] for d in
                 m78.lap_kbest(nen[c.id], master, oracle=v_dung[c.id])], dung)
        print(f"{'oracle (100 dòng vào video đúng)':<33}"
              f"{tong / len(giu):>9.4f}{tong / len(giu) - moc:>+9.4f}")
        print()


if __name__ == "__main__":
    main()
