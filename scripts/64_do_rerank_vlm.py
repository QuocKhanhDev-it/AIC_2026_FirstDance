"""
64_do_rerank_vlm.py — VLM chấm lại lấy được bao nhiêu trong 33 điểm của A54?

    python scripts/64_do_rerank_vlm.py --diem index/diem_vlm.jsonl

ĐẦU VÀO: file điểm do máy có GPU sinh ra (xem `notebooks/kaggle_rerank_vlm.md`)

    {"id": ..., "row_id": [..], "diem_vlm": [..], "model": "...", "dau": 30}

A54 đo khoảng trống giữa điểm thật và trần "xếp lại hoàn hảo" là **33 điểm
phần trăm**, và A55 cho thấy **không** lấy được bằng tín hiệu sẵn có. Đây là
phép đo xem thông tin MỚI (nhìn lại bức ảnh) lấy được bao nhiêu.

BỐN CÁCH DÙNG ĐIỂM VLM — vì "có điểm VLM" chưa nói được dùng thế nào

  1. thay hẳn thứ tự trong phần đầu bằng điểm VLM
  2. RRF giữa hạng gốc và hạng VLM (mấy trọng số)
  3. nhân: điểm gốc × (1 + w × điểm VLM chuẩn hoá)
  4. chỉ đẩy lên, không đẩy xuống: ứng viên VLM chấm cao được cộng, thấp giữ
     nguyên — tránh VLM kéo tụt câu kênh 1 vốn đã làm đúng

⚠️ PHẦN ĐUÔI GIỮ NGUYÊN. VLM chỉ chấm `dau` ứng viên đầu; các ứng viên sau
phải giữ đúng thứ tự cũ và nằm SAU toàn bộ phần đã xếp lại. Trộn lẫn hai vùng
là so hai thang điểm khác nhau — đúng lỗi RRF sinh ra để tránh.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402
from schema import Candidate                          # noqa: E402

W3 = 0.5
K_RRF = 60


def _lai(c: Candidate, diem: float) -> Candidate:
    return Candidate(row_id=c.row_id, video_id=c.video_id,
                     frame_idx=c.frame_idx, score=diem, source=c.source,
                     meta=c.meta)


def _chuan(d: dict) -> dict:
    """Đưa điểm VLM về [0,1] TRONG TỪNG CÂU. Chuẩn hoá toàn cục là sai: mỗi
    câu có độ khó khác nhau nên thang điểm tuyệt đối không so được."""
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def thay_han(ds, vlm, dau):
    dau_ds, duoi = ds[:dau], ds[dau:]
    xep = sorted(dau_ds, key=lambda c: -vlm.get(c.row_id, -1e9))
    return [_lai(c, 1.0 - i / len(xep)) for i, c in enumerate(xep)] + duoi


def rrf_hang(ds, vlm, dau, w):
    dau_ds, duoi = ds[:dau], ds[dau:]
    theo_vlm = sorted(dau_ds, key=lambda c: -vlm.get(c.row_id, -1e9))
    h_vlm = {c.row_id: i + 1 for i, c in enumerate(theo_vlm)}
    diem = {c.row_id: 1.0 / (K_RRF + i + 1) + w / (K_RRF + h_vlm[c.row_id])
            for i, c in enumerate(dau_ds)}
    xep = sorted(dau_ds, key=lambda c: -diem[c.row_id])
    return [_lai(c, diem[c.row_id]) for c in xep] + duoi


def nhan(ds, vlm, dau, w, chi_day_len=False):
    dau_ds, duoi = ds[:dau], ds[dau:]
    ch = _chuan({c.row_id: vlm[c.row_id] for c in dau_ds if c.row_id in vlm})
    def he_so(c):
        v = ch.get(c.row_id, 0.5)
        return 1.0 + w * (max(v - 0.5, 0.0) * 2 if chi_day_len else v)
    xep = sorted(dau_ds, key=lambda c: -c.score * he_so(c))
    return [_lai(c, c.score * he_so(c)) for c in xep] + duoi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--diem", default=GOC / "index" / "diem_vlm.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=300)
    a = ap.parse_args()

    if not a.diem.exists():
        raise SystemExit(
            f"Chưa có {a.diem}. Chạy `scripts/63_xuat_be_rerank.py` rồi làm "
            f"theo `notebooks/kaggle_rerank_vlm.md` trên máy có GPU.")

    vlm, dau, ten_model = {}, None, None
    for l in a.diem.read_text("utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        vlm[d["id"]] = dict(zip(d["row_id"], d["diem_vlm"]))
        dau = d.get("dau", len(d["row_id"]))
        ten_model = d.get("model", "?")
    print(f"điểm VLM: {len(vlm)} câu × {dau} ứng viên | model {ten_model}")

    master = pd.read_parquet(a.index / "master.parquet")
    cau = [c for c in tap_dev.doc(a.file) if c.id in vlm]

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)} câu\n")

    tho = {}

    def bo(c):
        if c.id not in tho:
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
            tho[c.id] = hop_nhat([anh, k3.tim(c.cau_hoi, k=a.be)],
                                 trong_so=[1.0, W3])
        return tho[c.id]

    cau_hinh = {
        "1. mốc: run.py (không VLM)": lambda c: bo(c)[:100],
        "2. VLM thay hẳn phần đầu": lambda c: thay_han(bo(c), vlm[c.id], dau)[:100],
    }
    for w in (0.5, 1.0, 2.0):
        cau_hinh[f"3. RRF hạng, VLM w={w:g}"] = (
            lambda w: lambda c: rrf_hang(bo(c), vlm[c.id], dau, w)[:100])(w)
    for w in (0.5, 1.0):
        cau_hinh[f"4. nhân, w={w:g}"] = (
            lambda w: lambda c: nhan(bo(c), vlm[c.id], dau, w)[:100])(w)
    cau_hinh["5. nhân, CHỈ đẩy lên (w=1)"] = lambda c: nhan(
        bo(c), vlm[c.id], dau, 1.0, chi_day_len=True)[:100]

    print(bao_cao_do_nhay(giu, cau_hinh, master))
    print("\nSo với A54: khoảng trống ở bể 300 là +0,3067 (±2s). Lấy được bao "
          "nhiêu phần trăm của con số đó mới là kết quả thật.")


if __name__ == "__main__":
    main()
