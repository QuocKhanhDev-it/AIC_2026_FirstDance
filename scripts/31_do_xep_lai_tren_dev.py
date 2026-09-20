"""
31_do_xep_lai_tren_dev.py — Việc 3 của 12_viec_cho_may_manh.md: đo trên TẬP DEV
kỹ thuật đã đưa điểm leaderboard 3,8 -> 5,4 (Gemini xếp lại top-50 bằng OCR/ASR),
thứ CHƯA BAO GIỜ được đo bằng `cham_diem.bao_cao_do_nhay` — chỉ dò thẳng bằng
leaderboard public (chấm 50% đáp án, lệch với xếp hạng cuối).

Import ĐỘNG `scripts/28_xep_lai_bang_gemini.py` để dùng ĐÚNG hàm `xep_lai()` +
`NHAC` đang chạy trong bài nộp thật — không chép lại, tránh đo một phiên bản
khác với cái đang dùng.

Mốc nền: SigLIP2 top-100 một mình (`cham_diem.diem_cau` đã chấm).
Cấu hình so: SigLIP2 top-100, Gemini xếp lại top-50 (đúng cấu hình đang dùng).

Chỉ đo câu KIS — kỹ thuật này (A27/A28) đo trên "top-N ứng viên KIS", QA/TRAKE
đọc theo cấu trúc bài nộp khác, ngoài phạm vi câu hỏi đang cần trả lời.

Gemini gọi ĐÚNG MỘT LẦN mỗi câu (không phải mỗi mức dung sai) — xếp hạng được
tính trước cho cả hai cấu hình rồi mới đưa vào `bao_cao_do_nhay`.

    python scripts/31_do_xep_lai_tren_dev.py --cache index/truy_van.npz
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
from bm25 import KenhVanBan                            # noqa: E402
from cham_diem import bao_cao_do_nhay                  # noqa: E402
from dense import KenhAnh, KenhAnhCache                # noqa: E402
from rrf import hop_nhat                               # noqa: E402
from schema import Candidate                           # noqa: E402


def _nap_script_28():
    """Nạp scripts/28_xep_lai_bang_gemini.py như một module — tên file bắt đầu
    bằng số nên không `import` được bằng cú pháp thường."""
    p = GOC / "scripts" / "28_xep_lai_bang_gemini.py"
    spec = importlib.util.spec_from_file_location("xep_lai_gemini", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--cache", type=Path, default=None,
                    help="index/truy_van.npz — không nạp model SigLIP2")
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--top", type=int, default=50,
                    help="đúng cấu hình đang dùng: top-50 (A28)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cho", type=float, default=4.5,
                    help="giây nghỉ giữa hai lượt gọi Gemini — free tier")
    ap.add_argument("--rrf-w", type=float, default=None,
                    help="thêm cấu hình thứ ba: RRF(kênh1,kênh3,w) rồi mới "
                         "xếp lại — kiểm A30 (bể ứng viên) cộng dồn với xếp "
                         "lại có tốt hơn xếp lại một mình không. Gọi Gemini "
                         "THÊM một lượt/câu (tổng ~2x)")
    a = ap.parse_args()

    xlg = _nap_script_28()
    model = a.model or xlg.MODEL_GEMINI

    index = GOC / "index"
    master = pd.read_parquet(index / "master.parquet")
    bang = pd.read_parquet(index / "ocr_asr.parquet")
    video_idx = {(r.video_id, int(r.frame_idx)): int(r.row_id)
                 for r in master.itertuples()}

    if a.cache:
        print(f"Kênh 1 từ cache {a.cache} — KHÔNG nạp model")
        k1 = KenhAnhCache(str(index), a.cache, matrix=a.matrix)
    else:
        k1 = KenhAnh(str(index), matrix=a.matrix)

    k3 = None
    if a.rrf_w is not None:
        k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")
        print(f"kênh 3 (OCR/ASR) nạp xong, {len(k3):,} khung — dùng cho "
              f"RRF(1,3,w={a.rrf_w})")

    cau = [c for c in tap_dev.doc() if c.loai == "KIS"]
    so_luot = len(cau) * (2 if a.rrf_w is not None else 1)
    print(f"{len(cau)} câu KIS | xếp lại top-{a.top} bằng {model}, "
          f"nghỉ {a.cho}s/lượt | ~{so_luot} lượt gọi Gemini\n")

    def _rerank(dong, c):
        moi, chon = xlg.xep_lai(dong, c.cau_hoi, bang, master, a.top, model)
        cand = [
            Candidate(row_id=video_idx[(v, int(f))], video_id=v,
                      frame_idx=int(f), score=0.0, source="xep_lai")
            for v, f in moi
        ]
        return cand, chon

    goc, xep_lai_kq, rrf_xep_lai_kq = {}, {}, {}
    for i, c in enumerate(cau, 1):
        kq = k1.tim(R.tach_truy_van(c.cau_hoi), k=100)
        goc[c.id] = kq

        dong = [[cd.video_id, cd.frame_idx] for cd in kq]
        xep_lai_kq[c.id], chon = _rerank(dong, c)
        msg = f"  [{i}/{len(cau)}] {c.id}  don:{len(chon)}"

        if a.rrf_w is not None:
            time.sleep(a.cho)
            kq3 = k3.tim(R.tach_truy_van(c.cau_hoi), k=100)
            rrf = hop_nhat([kq, kq3], trong_so=[1.0, a.rrf_w])[:100]
            dong_rrf = [[cd.video_id, cd.frame_idx] for cd in rrf]
            rrf_xep_lai_kq[c.id], chon2 = _rerank(dong_rrf, c)
            msg += f"  rrf+don:{len(chon2)}"

        print(msg)
        if i < len(cau):
            time.sleep(a.cho)

    cau_hinh = {
        "SigLIP2 top-100 (mốc nền)": lambda c: goc[c.id],
        f"+ Gemini xếp lại top-{a.top}": lambda c: xep_lai_kq[c.id],
    }
    if a.rrf_w is not None:
        cau_hinh[f"RRF(1,3,w={a.rrf_w}) + Gemini xếp lại top-{a.top}"] = \
            lambda c: rrf_xep_lai_kq[c.id]

    print("\n" + "=" * 70)
    print(bao_cao_do_nhay(cau, cau_hinh, master))


if __name__ == "__main__":
    main()
