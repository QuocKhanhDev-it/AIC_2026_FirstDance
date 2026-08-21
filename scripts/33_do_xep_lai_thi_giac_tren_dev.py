"""
33_do_xep_lai_thi_giac_tren_dev.py — Việc 6 của 12_viec_cho_may_manh.md: đo
trên TẬP DEV kỹ thuật "xếp lại bằng ảnh thật" (`scripts/30_xep_lai_thi_giac.py`),
lần này trên máy có ĐỦ 177.321/177.321 keyframe (100% kho) — không phải máy cũ
chỉ có 21% (A28).

A28 đo trên bài nộp: 5,4 -> 5,2, nhưng nghi phạm là HIỆN VẬT CỦA MÁY (chỉ nhìn
được ảnh của máy nào đã tải nhóm nào), không phải của kỹ thuật. Máy này đủ ảnh
toàn kho nên đo được sạch, không còn ẩn số đó.

Import ĐỘNG `scripts/30_xep_lai_thi_giac.py` để dùng ĐÚNG hàm `xep_lai()` đang
chạy trong bài nộp thật.

    python scripts/33_do_xep_lai_thi_giac_tren_dev.py --cache index/truy_van.npz
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
import tap_dev                                         # noqa: E402
from cham_diem import bao_cao_do_nhay                  # noqa: E402
from dense import KenhAnh, KenhAnhCache                # noqa: E402
from schema import Candidate                           # noqa: E402


def _nap_script_30():
    p = GOC / "scripts" / "30_xep_lai_thi_giac.py"
    spec = importlib.util.spec_from_file_location("xep_lai_thi_giac", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--so-anh", type=int, default=20)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cho", type=float, default=5.0)
    a = ap.parse_args()

    xlg = _nap_script_30()
    model = a.model or xlg.MODEL_GEMINI

    index = GOC / "index"
    master = pd.read_parquet(index / "master.parquet")
    video_idx = {(r.video_id, int(r.frame_idx)): int(r.row_id)
                 for r in master.itertuples()}
    co_anh = master.kf_path.notna().sum()
    print(f"Ảnh trên máy này: {co_anh:,}/{len(master):,} "
          f"({100 * co_anh / len(master):.1f}% kho)\n")

    if a.cache:
        print(f"Kênh 1 từ cache {a.cache} — KHÔNG nạp model")
        k1 = KenhAnhCache(str(index), a.cache, matrix=a.matrix)
    else:
        k1 = KenhAnh(str(index), matrix=a.matrix)

    cau = [c for c in tap_dev.doc() if c.loai == "KIS"]
    print(f"{len(cau)} câu KIS | xếp lại (ẢNH) top-{a.top}, tối đa "
          f"{a.so_anh} ảnh/câu, bằng {model}, nghỉ {a.cho}s/lượt\n")

    goc, xep_lai_kq = {}, {}
    for i, c in enumerate(cau, 1):
        kq = k1.tim(R.tach_truy_van(c.cau_hoi), k=100)
        goc[c.id] = kq

        dong = [[cd.video_id, cd.frame_idx] for cd in kq]
        moi, chon, n_anh = xlg.xep_lai(dong, c.cau_hoi, master, a.top,
                                        a.so_anh, model)
        xep_lai_kq[c.id] = [
            Candidate(row_id=video_idx[(v, int(f))], video_id=v,
                      frame_idx=int(f), score=0.0, source="xep_lai_thi_giac")
            for v, f in moi
        ]
        print(f"  [{i}/{len(cau)}] {c.id}  nhìn {n_anh:>2} ảnh  "
              f"đẩy lên {len(chon)}")
        if i < len(cau):
            time.sleep(a.cho)

    print("\n" + "=" * 70)
    print(bao_cao_do_nhay(
        cau,
        {
            "SigLIP2 top-100 (mốc nền)": lambda c: goc[c.id],
            f"+ Gemini xếp lại (ẢNH) top-{a.top}": lambda c: xep_lai_kq[c.id],
        },
        master,
    ))


if __name__ == "__main__":
    main()
