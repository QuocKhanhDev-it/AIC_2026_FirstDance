"""
98_do_bu_dong_trake.py — TRAKE: được cấp 100 dòng, K-best chỉ nộp một phần.

    .venv\\Scripts\\python.exe scripts\\98_do_bu_dong_trake.py

QUAN SÁT DẪN TỚI PHÉP ĐO NÀY

Chạy `run.py` trên `de_thi_thu` (cấu hình mặc định, sau A79) đếm được:

    query-p1-16-trake    58/100 dòng
    query-p1-4-trake     37/100 dòng
    query-p1-18-trake    11/100 dòng   <- vứt 89 chỗ

`kbest_trake.cham_video()` LOẠI mọi video thiếu ứng viên cho **bất kỳ** sự
kiện nào, nên số video sinh được chuỗi có thể rất nhỏ; `beam_video` còn lọc đa
dạng nữa. Cách CŨ (`run.dung_trake`) có nhánh bù rải video còn lại cho tròn
100 — cách mới bỏ mất lưới đó, mà phần bị bỏ thì chưa ai đo.

Theo chính PHẦN C mục 1 (*"không có điểm phạt, dòng thứ 100 vẫn đáng 0,2"*),
89 dòng trống là 89 cơ hội vứt đi. Nhưng đó là LẬP LUẬN — script này đo.

BA CẤU HÌNH

    K-best (mặc định hiện tại)  ← MỐC NỀN, đúng luật "mốc nền là cấu hình
                                MẠNH NHẤT hiện có"
    K-best + bù MỀM             video THIẾU sự kiện vẫn dựng chuỗi; chỗ thiếu
                                nội suy giữa hai neo (đúng cách `dung_trake`
                                điền chỗ trống, không bịa `khung + 1`)
    K-best + bù MỀM + RẢI       hết video có ứng viên thì rải đều trên khoảng
                                frame của các video còn lại

Bù CHỈ THÊM VÀO ĐUÔI, không đụng tới các dòng K-best đã sinh. Nên nếu vô ích
thì cùng lắm là hoà — không thể làm hỏng phần đang chạy được. Đó là lý do đáng
đo nó trước những ý tưởng đắt hơn.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                             # noqa: E402
import tap_dev                                              # noqa: E402
from bm25 import KenhVanBan                                 # noqa: E402
from cham_diem import bao_cao_tu_bang, cham_trake_nhieu_muc  # noqa: E402
from dense import KenhAnhCache                              # noqa: E402
from kbest_trake import (TOI_DA_UV, beam_video, cham_video,  # noqa: E402
                         gom_theo_video, lap_dong)
from rrf import hop_nhat                                    # noqa: E402

W3 = 0.5


def _noi_suy(cho, lo, hi, n):
    """Điền các vị trí `None` trong `cho`, rồi ép tăng thật.

    Cùng luật với `run.dung_trake` bước 2: bịa cho CÓ NGHĨA (nội suy giữa hai
    neo) chứ không bịa `khung_trước + 1` — ba sự kiện cách nhau 0,02 giây thì
    chắc chắn trượt.
    """
    co = [(i, v) for i, v in enumerate(cho) if v is not None]
    if not co:
        return None
    ra = list(cho)
    for i, v in enumerate(ra):
        if v is not None:
            continue
        truoc = [(j, x) for j, x in co if j < i]
        sau = [(j, x) for j, x in co if j > i]
        if truoc and sau:
            (j0, x0), (j1, x1) = truoc[-1], sau[0]
            ra[i] = x0 + round((x1 - x0) * (i - j0) / (j1 - j0))
        elif truoc:
            ra[i] = min(hi, truoc[-1][1] + round(
                (hi - truoc[-1][1]) * (i - truoc[-1][0]) / n))
        else:
            ra[i] = max(lo, sau[0][1] - round(
                (sau[0][1] - lo) * (sau[0][0] - i) / n))
    for i in range(1, n):
        if ra[i] <= ra[i - 1]:
            ra[i] = ra[i - 1] + 1
    return ra


def lap_bu(cac_su_kien, master, so_dong=100, rai=False):
    """`lap_dong` rồi BÙ cho tròn `so_dong`.

    Trả về hỗn hợp có chủ ý: dòng K-best mang `row_id` thật, dòng bù mang
    `(video_id, frame_idx)` — người gọi quy cả hai về `set` row_id, đúng cách
    `cham_diem.diem_trake_bai_nop` vốn nhận cả hai dạng (A5.7: 614 keyframe
    dùng chung `frame_idx`, đi qua `frame_idx` là mất thông tin).
    """
    ra = list(lap_dong(cac_su_kien, master, so_dong))
    if len(ra) >= so_dong:
        return ra[:so_dong]

    n = len(cac_su_kien)
    pts = master.pts_time.values
    fx = master.frame_idx.values
    theo_video = gom_theo_video(cac_su_kien)
    du = set(cham_video(theo_video))          # video ĐỦ sự kiện — đã dùng rồi

    # Điểm MỀM: chỉ cộng những sự kiện video này CÓ, cộng thưởng theo TỶ LỆ
    # phủ — video phủ 3/4 sự kiện đáng tin hơn video phủ 1/4.
    thieu = []
    for v, uv in theo_video.items():
        if v in du:
            continue
        co = [i for i, x in enumerate(uv) if x]
        if not co:
            continue
        thieu.append((len(co) / n
                      + sum(max(s for _, s in uv[i]) for i in co) / len(co), v))
    thieu.sort(reverse=True)

    bien = master.groupby("video_id").frame_idx.agg(["min", "max"])
    da_dung = set()
    for _, v in thieu:
        if len(ra) >= so_dong:
            break
        uv = theo_video[v]
        # Neo = ứng viên tốt nhất của từng sự kiện, bỏ neo phá thứ tự thời gian.
        neo, cuoi = [], -1e9
        for x in uv:
            r = max(x, key=lambda t: t[1])[0] if x else None
            if r is not None and pts[r] > cuoi:
                neo.append(int(fx[r]))
                cuoi = pts[r]
            else:
                neo.append(None)
        kh = _noi_suy(neo, int(bien.loc[v, "min"]), int(bien.loc[v, "max"]), n)
        if kh:
            ra.append([(v, f) for f in kh])
            da_dung.add(v)

    if rai:
        con = [v for v in bien.index if v not in da_dung and v not in du]
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
            ra.append([(v, f) for f in kh])
    return ra[:so_dong]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl",
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

    tra = {}
    for r_, v_, f_ in zip(master.row_id.values, master.video_id.values,
                          master.frame_idx.values):
        tra.setdefault((str(v_), int(f_)), set()).add(int(r_))

    nen = {}
    for c in giu:
        cac = []
        for sk in R.tach_su_kien(c.cau_hoi):
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(sk)])
            cac.append(hop_nhat([anh, k3.tim(sk, k=a.be)], trong_so=[1.0, W3]))
        nen[c.id] = cac

    def quy(dong):
        """Dòng -> `list[list[set row_id]]`; nhận cả row_id lẫn (video, frame)."""
        return [[(tra.get(x, set()) if isinstance(x, tuple) else {x}) for x in d]
                for d in dong]

    print(f"{'câu':<20}{'K-best':>8}{'+bù mềm':>10}{'+bù rải':>10}")
    tong = [0, 0, 0]
    for c in giu:
        s = [len(lap_dong(nen[c.id], master)), len(lap_bu(nen[c.id], master)),
             len(lap_bu(nen[c.id], master, rai=True))]
        tong = [x + y for x, y in zip(tong, s)]
        print(f"{c.id:<20}{s[0]:>8}{s[1]:>10}{s[2]:>10}")
    n = len(giu)
    print(f"{'TRUNG BÌNH':<20}" + "".join(f"{x / n:>10.1f}" for x in tong) + "\n")

    def nho(f):
        c_ = {}

        def g(c):
            if c.id not in c_:
                c_[c.id] = f(c)
            return c_[c.id]
        return g

    cau_hinh = {
        "K-best ← MỐC": nho(lambda c: quy(lap_dong(nen[c.id], master))),
        "+ bù MỀM": nho(lambda c: quy(lap_bu(nen[c.id], master))),
        "+ bù MỀM + RẢI": nho(lambda c: quy(lap_bu(nen[c.id], master, rai=True))),
    }
    bang = {t: cham_trake_nhieu_muc(giu, f, master) for t, f in cau_hinh.items()}
    print(bao_cao_tu_bang(bang))


if __name__ == "__main__":
    main()
