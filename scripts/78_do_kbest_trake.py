"""
78_do_kbest_trake.py — Dồn 100 dòng TRAKE vào vài video, thay vì 1 dòng/video.

    python scripts/78_do_kbest_trake.py

VẤN ĐỀ (A63): kênh tìm được 0,3500 mà bài nộp chỉ 0,1667 — mất 52%.

Nguyên nhân đã soi ra: `dung_trake()` xếp **một dòng cho mỗi video**, rồi bù
cho đủ 100 bằng 99 video xếp sau. Nhưng TRAKE chấm THEO VỊ TRÍ: chuỗi của video
tốt nhất lệch một sự kiện ra ngoài cửa sổ là **cả dòng đó 0 điểm**, và 99 dòng
còn lại dành cho những video gần như chắc chắn sai.

Với TRAKE, toàn bộ đáp án nằm trong MỘT video. Nên 100 dòng nên là **100 giả
thuyết khác nhau về video đúng**, không phải 100 video khác nhau.

CÁCH LÀM

  1. Chấm điểm video: Σ log(điểm cao nhất của video cho từng sự kiện). Video
     thiếu hẳn ứng viên cho một sự kiện nào đó -> loại.
  2. Lấy top-M video, chia ngân sách 100 dòng theo thứ hạng.
  3. Trong mỗi video, **beam search** sinh K chuỗi KHÁC NHAU, đều tăng dần
     ngặt theo thời gian. Phạt chuỗi trùng thời điểm để K chuỗi trải ra chứ
     không dính vào một giây.

⚠️ KHÁC HẲN "ràng buộc đa dạng" đã bị bác ở A61 (mỗi video tối đa 3 dòng,
−0,0827). Cái đó áp cho KIS, nơi đáp án có thể ở bất kỳ video nào nên cắt bớt
video là cắt mất cơ hội. TRAKE thì ngược lại: đáp án CHẮC CHẮN trong một video.

DÒNG "TRẦN" TRONG BẢNG

`oracle` dồn cả 100 dòng vào ĐÚNG video chứa đáp án. Không dùng được lúc thi —
nó trả lời "nếu chọn đúng video thì beam search moi được bao nhiêu", tách phần
CHỌN VIDEO khỏi phần LẮP CHUỖI.
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import (DUNG_SAI_CHINH, DUNG_SAI_KIEM,  # noqa: E402
                       MOC, bao_cao_tu_bang, cham_trake_nhieu_muc,
                       no_cua_so)
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
BEAM = 64
CACH_NHAU = [1.5]        # giây — hai chuỗi coi là khác nhau nếu lệch quá ngần
                         # này. Để dạng list cho `--cach-nhau` ghi đè được:
                         # cửa sổ ±2s cần chuỗi mịn hơn cửa sổ ±15s.


def diem_bai_nop(cac_dong, dung_moi_su_kien, gioi_han=100):
    """R-Score theo cách BTC chấm: vị trí i chỉ so với sự kiện i."""
    n = len(dung_moi_su_kien)
    if not n:
        return 0.0
    r = [sum(1 for i, tap in enumerate(d[:n]) if tap & dung_moi_su_kien[i]) / n
         for d in cac_dong[:gioi_han]]
    return sum(max(r[:k], default=0.0) for k in MOC) / len(MOC) if r else 0.0


def beam_video(uv_theo_su_kien, pts, k_chuoi):
    """Sinh tối đa `k_chuoi` chuỗi tăng dần ngặt, khác nhau về thời gian.

    `uv_theo_su_kien[i]` = [(row_id, điểm)] của sự kiện i TRONG một video.
    """
    beam = [([], 0.0, -1e9)]                       # (chuỗi, điểm, pts cuối)
    for uv in uv_theo_su_kien:
        moi = []
        for chuoi, d, t_cuoi in beam:
            for rid, s in uv:
                if pts[rid] > t_cuoi:
                    moi.append((chuoi + [rid], d + s, pts[rid]))
        if not moi:                                # không nối tiếp được nữa
            return [c for c, _, _ in beam if len(c) == len(uv_theo_su_kien)]
        moi.sort(key=lambda x: -x[1])
        beam = moi[:BEAM]

    # Đa dạng hoá: bỏ chuỗi quá giống chuỗi đã giữ (mọi vị trí đều sát nhau).
    ra = []
    for chuoi, _, _ in beam:
        if all(any(abs(pts[a] - pts[b]) > CACH_NHAU[0]
                   for a, b in zip(chuoi, cu)) for cu in ra):
            ra.append(chuoi)
        if len(ra) >= k_chuoi:
            break
    return ra


def lap_kbest(cac, master, so_dong=100, so_video=5, oracle=None):
    """Trả về danh sách dòng, mỗi dòng là list row_id theo thứ tự sự kiện."""
    pts = master.pts_time.values
    vid = master.video_id.values
    n = len(cac)

    theo_video = {}
    for i, ds in enumerate(cac):
        for c in ds:
            theo_video.setdefault(c.video_id, [[] for _ in range(n)])[i].append(
                (c.row_id, c.score))

    diem_v = {}
    for v, uv in theo_video.items():
        if any(not x for x in uv):                 # thiếu sự kiện -> loại
            continue
        diem_v[v] = sum(math.log(max(s for _, s in x) + 1e-9) for x in uv)
    if not diem_v:
        return []

    xep = ([oracle] if oracle and oracle in theo_video
           else sorted(diem_v, key=lambda v: -diem_v[v])[:so_video])
    # Ngân sách giảm dần: video đầu ăn nửa số dòng.
    ngan = ([so_dong] if oracle else
            [max(1, round(so_dong * w)) for w in (0.5, 0.25, 0.15, 0.07, 0.03)])

    ra = []
    for v, k in zip(xep, ngan):
        uv = theo_video[v]
        if any(not x for x in uv):
            continue
        for x in uv:
            x.sort(key=lambda t: -t[1])
        ra += beam_video([x[:20] for x in uv], pts, k)
        if len(ra) >= so_dong:
            break
    return ra[:so_dong]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--cach-nhau", type=float, nargs="*", default=[0.5, 1.5, 3.0],
                    help="các mức giãn thời gian giữa hai chuỗi, đem dò")
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
        tra.setdefault((v_, int(f_)), set()).add(int(r_))

    # Tính bể ứng viên MỘT LẦN cho mọi cấu hình.
    nen = {}
    for c in giu:
        cac = []
        for sk in R.tach_su_kien(c.cau_hoi):
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(sk)])
            cac.append(hop_nhat([anh, k3.tim(sk, k=a.be)], trong_so=[1.0, W3]))
        nen[c.id] = cac

    for ds_giay in (DUNG_SAI_CHINH, DUNG_SAI_KIEM):
        print(f"── dung sai ±{ds_giay:g}s " + "─" * 46)
        print(f"{'câu':<18}{'CŨ':>8}" +
              "".join(f"{'K-best ' + str(g) + 's':>13}" for g in a.cach_nhau) +
              f"{'oracle':>9}")
        tong = [0.0] * (len(a.cach_nhau) + 2)
        for c in giu:
            cac = nen[c.id]
            dung = [no_cua_so(b, master, ds_giay) for b in c.row_id_dung]
            cu = [[tra.get((d.video_id, f), set()) for f in d.frame_idxs]
                  for d in R.dung_trake(cac, master)]
            hang = [diem_bai_nop(cu, dung)]
            for g in a.cach_nhau:
                CACH_NHAU[0] = g
                hang.append(diem_bai_nop(
                    [[{r} for r in d] for d in lap_kbest(cac, master)], dung))
            CACH_NHAU[0] = a.cach_nhau[0]
            v_dung = master.video_id.iloc[c.row_id_dung[0][0]]
            hang.append(diem_bai_nop(
                [[{r} for r in d]
                 for d in lap_kbest(cac, master, oracle=v_dung)], dung))
            tong = [x + y for x, y in zip(tong, hang)]
            print(f"{c.id:<18}{hang[0]:>8.4f}"
                  + "".join(f"{x:>13.4f}" for x in hang[1:-1])
                  + f"{hang[-1]:>9.4f}")
        n = len(giu)
        print(f"{'TRUNG BÌNH':<18}{tong[0]/n:>8.4f}"
              + "".join(f"{x/n:>13.4f}" for x in tong[1:-1])
              + f"{tong[-1]/n:>9.4f}\n")

    print(bao_cao(giu, nen, master, a.cach_nhau))


def bao_cao(giu, nen, master, cach_nhau):
    """Bảng có NGƯỠNG NHIỄU và kết luận ✅/🟡/❌, như mọi phép đo khác.

    Trước đây script này chỉ in điểm trung bình. Hệ quả: A74 đo được K-best
    +0,1029 ở ±2s mà **không bật được**, vì không biết con số đó có vượt nhiễu
    hay không — chứ không phải vì nó sai.
    """
    tra = {}
    for r_, v_, f_ in zip(master.row_id.values, master.video_id.values,
                          master.frame_idx.values):
        tra.setdefault((v_, int(f_)), set()).add(int(r_))

    def nho(f):
        c_ = {}

        def g(c):
            if c.id not in c_:
                c_[c.id] = f(c)
            return c_[c.id]
        return g

    def cu_(c):
        return [[tra.get((d.video_id, f), set()) for f in d.frame_idxs]
                for d in R.dung_trake(nen[c.id], master)]

    def kbest(g):
        def f(c):
            CACH_NHAU[0] = g
            return lap_kbest(nen[c.id], master)
        return f

    # MỐC phải đứng ĐẦU — `bao_cao_tu_bang` lấy mục đầu tiên làm mốc nền.
    cau_hinh = {"CŨ 1 dòng/video ← MỐC": nho(cu_)}
    for g in cach_nhau:
        cau_hinh[f"K-best {g:g}s"] = nho(kbest(g))
    CACH_NHAU[0] = cach_nhau[0]

    bang = {ten: cham_trake_nhieu_muc(giu, f, master)
            for ten, f in cau_hinh.items()}
    return bao_cao_tu_bang(bang)


if __name__ == "__main__":
    main()
