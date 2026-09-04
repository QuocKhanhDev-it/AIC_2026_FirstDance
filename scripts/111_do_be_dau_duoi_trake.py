"""
111_do_be_dau_duoi_trake.py — Bể lớn cho ĐUÔI, bể nhỏ cho ĐẦU: tách hai vai trò của `--be`.

    python scripts/111_do_be_dau_duoi_trake.py

VÌ SAO CÓ PHÉP ĐO NÀY

A94 (`110_`) đo nới bể ứng viên mỗi sự kiện 100 -> 300 được **+0,0739 ở ±2s /
+0,0533 ở ±15s**, 7 thắng 3 thua. Nhưng nhìn từng câu thì hiệu ứng **dồn vào
hai câu**:

    trake-L25-004   0,0000 -> 0,4800   (+0,4800)
    trake-DE2-21    0,0000 -> 0,5000   (+0,5000)
    -> hai câu chiếm 74% tổng hiệu

và **ba câu bị hại**, câu nặng nhất mất −0,2000 (0,9500 -> 0,7500).

Hai chiều đó có cơ chế khác nhau, và đó là lý do tách được:

* **Được:** bể lớn làm video ĐÚNG lọt vào danh sách, ở những câu mà trước đây
  nó không có mặt -> câu 0 điểm thành có điểm. Chuyện này xảy ra ở **đuôi**
  danh sách (hạng 6-25, mỗi video 1 dòng).
* **Mất:** bể lớn cũng thả video NHIỄU vào, và chúng tranh **hạn ngạch top-5**
  (40/25/15/12/8 = 80 dòng). Một video đúng đang ở hạng 2 bị đẩy xuống hạng 4
  là mất 13 dòng.

Nếu tách đúng thì lấy được vế đầu mà không trả vế sau.

BỐN CẤU HÌNH — LƯỚI 2×2 THẬT, KHÔNG PHẢI BA BIẾN THỂ

    ĐẦU (top-5)   ĐUÔI (hạng 6-25)
    bể 100        bể 100        <- 1. MỐC, đang chạy
    bể 300        bể 300        <- 2. A94, cả hai cùng nới
    bể 100        bể 300        <- 3. GIẢ THUYẾT: chỉ nới đuôi
    bể 300        bể 100        <- 4. ĐỐI CHỨNG: chỉ nới đầu

Dòng 4 là nhóm đối chứng và nó bắt buộc phải có: nếu dòng 3 và dòng 4 cùng
thắng như nhau thì giả thuyết "được ở đuôi, mất ở đầu" SAI, và cái thắng đến
từ chỗ khác.

⚠️ MỘT DÒNG ĐUÔI ĐỦ ĐỂ ĐƯỢC 0,48. Điểm = trung bình `max R-Score trong top-k`
trên k ∈ {1,5,20,50,100}. Một dòng duy nhất ở hạng 6-20 với R-Score 0,6 (khớp
3/5 sự kiện) cho `mean(0; 0,6; 0,6; 0,6; 0,6) = 0,48` — đúng bằng con số
`trake-L25-004` nhận được. Nên giả thuyết "đuôi làm nên chuyện" là khả dĩ về
mặt số học, không chỉ về mặt kể chuyện.

⚠️ Chấm ở tầng NỘP.
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
from cham_diem import bao_cao_tu_bang                 # noqa: E402
from cham_diem import cham_trake_nhieu_muc            # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

import kbest_trake as KB                              # noqa: E402

W3 = 0.5
BE_NHO, BE_LON = 100, 300


def lap_dong_dau_duoi(ds_dau, ds_duoi, master, so_dong: int = 100):
    """Như `kbest_trake.lap_dong` nhưng ĐẦU và ĐUÔI dùng hai bể khác nhau.

    Chỉ đổi nguồn ứng viên; hạn ngạch 40/25/15/12/8, 20 dòng đuôi, beam, giãn
    cách đều giữ nguyên — nếu không thì thắng thua không quy được cho ai.
    """
    pts = master.pts_time.values

    def xep_va_chuoi(ds):
        tv = KB.gom_theo_video(ds)
        d = KB.cham_video(tv)

        def chuoi(v, k):
            uv = [list(x) for x in tv[v]]
            for x in uv:
                x.sort(key=lambda t: -t[1])
            return KB.beam_video([x[:KB.TOI_DA_UV] for x in uv], pts, k,
                                 KB.CACH_NHAU)
        return sorted(d, key=lambda v: -d[v]), chuoi

    xep_d, chuoi_d = xep_va_chuoi(ds_dau)
    xep_t, chuoi_t = xep_va_chuoi(ds_duoi)
    if not xep_d:
        xep_d, chuoi_d = xep_t, chuoi_t

    n_tren = max(1, so_dong - KB.N_DUOI)
    ra = []
    for v, w in zip(xep_d[:KB.SO_VIDEO], KB.TY_LE):
        ra += chuoi_d(v, max(1, round(n_tren * w)))
        if len(ra) >= n_tren:
            break
    ra = ra[:n_tren]

    # Đuôi lấy từ bể riêng, BỎ QUA những video đã dùng ở đầu — nếu không thì
    # bể lớn chỉ lặp lại chính năm video kia và phép đo không đo gì cả.
    da_dung = set(xep_d[:KB.SO_VIDEO])
    con = [v for v in xep_t if v not in da_dung]
    for v in con[:KB.N_DUOI]:
        ra += chuoi_t(v, 1)
    return ra[:so_dong]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl",
        GOC / "dev" / "tap_dev_trake.jsonl"])
    ap.add_argument("--be-nho", type=int, default=BE_NHO)
    ap.add_argument("--be-lon", type=int, default=BE_LON)
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
    print(f"\n{len(giu)} câu TRAKE | bể nhỏ {a.be_nho}, bể lớn {a.be_lon}\n")

    nho = {}

    def ung_vien(c, be):
        if (c.id, be) not in nho:
            ds = []
            for sk in R.tach_su_kien(c.cau_hoi):
                anh = hop_nhat([k1.tim(m, k=be) for m in R.tach_truy_van(sk)])
                ds.append(hop_nhat([anh, k3.tim(sk, k=be)],
                                   trong_so=[1.0, W3])[:be])
            nho[(c.id, be)] = ds
        return nho[(c.id, be)]

    def lam(be_dau, be_duoi):
        return lambda c: lap_dong_dau_duoi(
            ung_vien(c, be_dau), ung_vien(c, be_duoi), master, 100)

    n, l = a.be_nho, a.be_lon
    cau_hinh = {
        f"1. MỐC: đầu {n} / đuôi {n}": lam(n, n),
        f"2. A94: đầu {l} / đuôi {l}": lam(l, l),
        f"3. GIẢ THUYẾT: đầu {n} / đuôi {l}": lam(n, l),
        f"4. ĐỐI CHỨNG: đầu {l} / đuôi {n}": lam(l, n),
    }
    bang = {k: cham_trake_nhieu_muc(giu, f, master)
            for k, f in cau_hinh.items()}
    print(bao_cao_tu_bang(bang))
    print("\nĐỌC BẢNG: nếu dòng 3 thắng mà dòng 4 KHÔNG, giả thuyết đúng —\n"
          "được là nhờ đuôi, mất là do đầu. Nếu cả hai cùng thắng như nhau\n"
          "thì giả thuyết SAI và cái thắng đến từ chỗ khác.")


if __name__ == "__main__":
    main()
