"""
22_do_trake.py — Đo TRAKE ở đúng tầng BTC chấm bài nộp, không phải tầng kênh.

`cham_diem.diem_trake()` chỉ chấm KÊNH ("có tìm ra sự kiện không"). Câu hỏi
thật là "nộp cái này thì được mấy điểm" — đó là việc của `run.dung_trake()`
(lắp ráp N khung thành một dòng nộp) và `cham_diem.diem_trake_bai_nop()`
(chấm đúng công thức BTC: mỗi VỊ TRÍ i chỉ so với sự kiện i). Hai tầng có thể
lệch xa: kênh tìm đủ N sự kiện nhưng lắp sai vị trí thì tầng dưới vẫn cho 0.

Đo hai thứ trên CÙNG 3 câu TRAKE của tập dev (n=3 — không đủ để kết luận
"ổn định" theo kỷ luật đo của dự án, chỉ dùng để kiểm KHÔNG TỆ ĐI rõ rệt):

  A. Bước 2b: xep_video_theo_chuoi (mềm) so với video_du_chuoi (giao cứng) —
     đã đổi trong run.py, đo để xác nhận không làm tệ đi.
  B. Bước 3: có nên trộn kênh 4 (objects) vào từng sự kiện bằng RRF không —
     CHƯA đổi trong run.py, đo trước khi quyết định.

    python scripts/22_do_trake.py
"""

import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from cham_diem import diem_trake_bai_nop, no_cua_so, MOC_DUNG_SAI  # noqa: E402
from dense import KenhAnh, KenhAnhCache               # noqa: E402
from rrf import hop_nhat                              # noqa: E402
import run as R                                       # noqa: E402
from run import tach_su_kien                          # noqa: E402
from thoi_gian import video_du_chuoi, xep_video_theo_chuoi  # noqa: E402

sys.path.insert(0, str(GOC / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("r16", GOC / "scripts" / "16_do_rrf.py")
_r16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r16)
KenhObjects = _r16.KenhObjects


def hang_thanh_row_id(dong_trake, master, video_idx: dict) -> list:
    """AnswerTRAKE (video_id, [frame_idx...]) -> [row_id...], để chấm bằng
    diem_trake_bai_nop (hàm đó so trên row_id, không so video_id+frame_idx)."""
    ra = []
    for f in dong_trake.frame_idxs:
        r = video_idx.get((dong_trake.video_id, f))
        ra.append(r if r is not None else -1)   # -1: khớp được, khỏi lẫn None
    return ra


def main():
    import argparse
    ap = argparse.ArgumentParser(description="do TRAKE")
    ap.add_argument("--cache", type=Path, default=None,
                    help="file .npz vector truy vấn đã mã hoá sẵn "
                         "(scripts/25_ma_hoa_truy_van.py). Có nó thì KHÔNG "
                         "phải nạp model — chạy được trên máy thiếu RAM")
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    a = ap.parse_args()

    index = GOC / "index"
    master = pd.read_parquet(index / "master.parquet")
    video_idx = {(r.video_id, int(r.frame_idx)): int(r.row_id)
                 for r in master.itertuples()}

    cau = [c for c in tap_dev.doc() if c.loai == "TRAKE"]
    print(f"{len(cau)} câu TRAKE — n QUÁ NHỎ để kết luận ổn định, "
          f"chỉ dùng để kiểm không tệ đi\n")

    if a.cache:
        print(f"Kênh 1 từ cache {a.cache} — KHÔNG nạp model")
        k1 = KenhAnhCache(str(index), a.cache, matrix=a.matrix)
        thieu = k1.co_du([c.cau_hoi for c in cau])
        if thieu:
            raise SystemExit(
                f"{len(thieu)} câu chưa có trong cache, ví dụ:\n"
                f"  {thieu[0][:100]!r}\n"
                f"Mã hoá thêm: python scripts/25_ma_hoa_truy_van.py "
                f"--tap-dev --gop")
    else:
        print("Nạp kênh 1 (SigLIP2) và kênh 4 (objects)...")
        k1 = KenhAnh(str(index), matrix=a.matrix)
    k4 = KenhObjects(index, master)

    cau_hinh = {
        "A_cu (video_du_chuoi + sorted() cu, kenh1 mot minh)": {
            "xep_video": video_du_chuoi, "dung_kenh4": False, "that": False},
        "B (xep_video_theo_chuoi + sorted() cu, kenh1 mot minh)": {
            "xep_video": xep_video_theo_chuoi, "dung_kenh4": False, "that": False},
        "C (xep_video_theo_chuoi + sorted() cu, RRF kenh1+4)": {
            "xep_video": xep_video_theo_chuoi, "dung_kenh4": True, "that": False},
        "D_THAT (run.dung_trake that su — DP + xep_video_theo_chuoi)": {
            "xep_video": None, "dung_kenh4": False, "that": True},
    }

    ket_qua = {ten: [] for ten in cau_hinh}
    for c in cau:
        su_kien_text = tach_su_kien(c.cau_hoi)
        n = len(su_kien_text)
        print(f"  {c.id}: {n} sự kiện")

        ds_k1 = [k1.tim(sk, k=100) for sk in su_kien_text]
        ds_k4 = [k4.tim(sk, k=100) for sk in su_kien_text]
        ds_rrf = [hop_nhat([a, b]) for a, b in zip(ds_k1, ds_k4)]

        # ⚠️ BTC chấm frame_idx rơi TRONG một khoảng, không cần trúng đúng một
        # row_id (PHẦN C mục 5). So chính xác tuyệt đối là chấm CHẶT HƠN BTC —
        # đúng bẫy `no_cua_so()` sinh ra để tránh. Báo cả hai mức dung sai.
        dung_theo_muc = {
            ds_giay: [no_cua_so(b, master, ds_giay) for b in c.row_id_dung]
            for ds_giay in MOC_DUNG_SAI
        }

        for ten, cfg in cau_hinh.items():
            ds = ds_rrf if cfg["dung_kenh4"] else ds_k1
            if cfg["that"]:
                dong = R.dung_trake(ds, master)
            else:
                dong = dung_trake_voi_xep_video(ds, master, cfg["xep_video"])
            cac_dong = [hang_thanh_row_id(d, master, video_idx) for d in dong]
            diems = {ds_giay: diem_trake_bai_nop(cac_dong, dung, gioi_han=100)
                     for ds_giay, dung in dung_theo_muc.items()}
            ket_qua[ten].append(diems)
            in_diem = "  ".join(f"±{k}s={v:.4f}" for k, v in diems.items())
            print(f"    {ten}: {in_diem}")

    print("\n" + "=" * 76)
    print("TỔNG KẾT (trung bình 3 câu — n=3, KHÔNG đủ để kết luận ổn định)")
    print("=" * 76)
    for ten, ds_list in ket_qua.items():
        for ds_giay in MOC_DUNG_SAI:
            vals = [d[ds_giay] for d in ds_list]
            print(f"  {ten:<55} ±{ds_giay}s: {sum(vals) / len(vals):.4f}   "
                  f"{['%.3f' % x for x in vals]}")


def dung_trake_voi_xep_video(cac_su_kien, master, ham_xep_video, so_dong=100):
    """Bản `run.dung_trake()` cho phép đổi hàm xếp video — để so A/B mà không
    sửa `run.py`. Sao chép logic lắp ráp (nội suy/ép tăng dần/bù cho đủ) y hệt
    bản gốc, chỉ khác nguồn `uu_tien`."""
    from schema import AnswerTRAKE
    n = len(cac_su_kien)
    if n == 0:
        return []
    uu_tien = ham_xep_video(cac_su_kien)
    if ham_xep_video is video_du_chuoi:
        for ds in cac_su_kien:
            for c in ds:
                if c.video_id not in uu_tien:
                    uu_tien.append(c.video_id)

    bien = master.groupby("video_id").frame_idx.agg(["min", "max"])
    DON_NHAU = 100
    ra = []
    for vid in uu_tien[:so_dong]:
        tot = []
        for ds in cac_su_kien:
            trong = [c for c in ds if c.video_id == vid]
            tot.append(int(trong[0].frame_idx) if trong else None)
        if all(x is None for x in tot):
            continue
        lo, hi = int(bien.loc[vid, "min"]), int(bien.loc[vid, "max"])
        co = [(i, v) for i, v in enumerate(tot) if v is not None]
        for i, v in enumerate(tot):
            if v is not None:
                continue
            truoc = [(j, x) for j, x in co if j < i]
            sau = [(j, x) for j, x in co if j > i]
            if truoc and sau:
                (j0, x0), (j1, x1) = truoc[-1], sau[0]
                tot[i] = x0 + round((x1 - x0) * (i - j0) / (j1 - j0))
            elif truoc:
                tot[i] = min(hi, truoc[-1][1] + round((hi - truoc[-1][1])
                                                      * (i - truoc[-1][0]) / n))
            else:
                tot[i] = max(lo, sau[0][1] - round((sau[0][1] - lo)
                                                   * (sau[0][0] - i) / n))
        khung = sorted(int(x) for x in tot)
        if n > 1 and khung[-1] - khung[0] < DON_NHAU and hi > lo:
            buoc = (hi - lo) / (n + 1)
            khung = [lo + round(buoc * (i + 1)) for i in range(n)]
        for i in range(1, n):
            if khung[i] <= khung[i - 1]:
                khung[i] = khung[i - 1] + 1
        ra.append(AnswerTRAKE(vid, khung))
        if len(ra) >= so_dong:
            break

    if len(ra) < so_dong:
        da_co = {x.video_id for x in ra}
        con = [v for v in bien.index if v not in da_co]
        buoc = max(1, len(con) // max(so_dong - len(ra), 1))
        for v in con[::buoc]:
            if len(ra) >= so_dong:
                break
            lo, hi = int(bien.loc[v, "min"]), int(bien.loc[v, "max"])
            b = (hi - lo) / (n + 1)
            kh = [lo + round(b * (i + 1)) for i in range(n)]
            for i in range(1, n):
                if kh[i] <= kh[i - 1]:
                    kh[i] = kh[i - 1] + 1
            ra.append(AnswerTRAKE(v, kh))
    return ra


if __name__ == "__main__":
    main()
