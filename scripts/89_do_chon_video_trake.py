"""
89_do_chon_video_trake.py — Chấm điểm VIDEO cho TRAKE: phạt mắt xích yếu tới đâu?

    python scripts/89_do_chon_video_trake.py

A74 đo được K-best beam search lấy lại +0,1029 ở ±2s, nhưng **khoảng cách tới
oracle vẫn 0,1029** — tức còn đúng ngần ấy nằm ở khâu CHỌN VIDEO, không phải
khâu lắp chuỗi. Bằng chứng cụ thể: `trake-L25-004` rơi 0,4800 -> 0,0000 ở CẢ
HAI mức dung sai dù oracle của nó là 0,8000.

GIẢ THUYẾT

Video đúng thường có **TẤT CẢ** sự kiện khớp ở mức chấp nhận được. Video nhiễu
thường chỉ có 1–2 sự kiện khớp tình cờ rất mạnh. Nên cách hợp điểm càng phạt
nặng **mắt xích yếu nhất** thì càng chọn đúng video.

BỐN CÁCH CHẤM XẾP THÀNH MỘT HỌ ĐƠN ĐIỆU — và đó là điều làm phép đo này đọc được

Gọi `m_e` = điểm cao nhất của video cho sự kiện `e`. Vì mọi video đều có cùng
số sự kiện, `Σ log(m_e)` **tương đương đơn điệu với trung bình NHÂN**. Nên:

    tổng      = trung bình CỘNG      <- ít khắt khe nhất (đối chứng)
    tổng-log  = trung bình NHÂN      <- ĐANG DÙNG
    điều hoà  = trung bình ĐIỀU HOÀ  <- khắt khe hơn
    min       = mắt xích yếu nhất    <- khắt khe nhất

Bất đẳng thức trung bình bảo đảm thứ tự đó. Nên kết quả nói được một điều dù đi
hướng nào: nếu điểm tăng đơn điệu theo độ khắt khe thì giả thuyết đúng và nên
đi tiếp; nếu nó hình chuông quanh trung bình nhân thì cấu hình hiện tại đã ở
đỉnh; nếu giảm đơn điệu thì giả thuyết sai chiều.

⚠️ ĐO HAI TẦNG. Điểm cuối lẫn lộn "chọn video" với "lắp chuỗi". Bảng đầu đo
thẳng thứ cần đo: **video đúng nằm ở hạng mấy** theo mỗi cách chấm.
"""

import argparse
import importlib.util
import math
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

# Nạp lại 78_ thay vì chép hàm sang đây — chép là hai bản sẽ trôi khỏi nhau.
_s = importlib.util.spec_from_file_location(
    "m78", GOC / "scripts" / "78_do_kbest_trake.py")
m78 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m78)

W3 = 0.5
DUNG_SAI = (2.0, 15.0)


def cong(m):        return sum(m)                       # trung bình cộng
def nhan(m):        return sum(math.log(x + 1e-9) for x in m)   # ĐANG DÙNG
def dieu_hoa(m):    return len(m) / sum(1.0 / (x + 1e-9) for x in m)
def nho_nhat(m):    return min(m)


CACH = {"1. tổng (cộng)": cong,
        "2. tổng-log (nhân) ← MỐC": nhan,
        "3. điều hoà": dieu_hoa,
        "4. min (mắt xích yếu)": nho_nhat}


def xep_video(cac, cham):
    """`{video_id: điểm}` theo cách chấm `cham`. Thiếu sự kiện nào thì loại."""
    theo_video = {}
    n = len(cac)
    for i, ds in enumerate(cac):
        for c in ds:
            theo_video.setdefault(c.video_id, [[] for _ in range(n)])[i].append(
                (c.row_id, c.score))
    ra = {}
    for v, uv in theo_video.items():
        if any(not x for x in uv):
            continue
        ra[v] = cham([max(s for _, s in x) for x in uv])
    return ra, theo_video


def lap(cac, master, cham, so_dong=100, so_video=5):
    """Y hệt `m78.lap_kbest` nhưng cách chấm VIDEO thay được."""
    pts = master.pts_time.values
    diem_v, theo_video = xep_video(cac, cham)
    if not diem_v:
        return []
    xep = sorted(diem_v, key=lambda v: -diem_v[v])[:so_video]
    ngan = [max(1, round(so_dong * w)) for w in (0.5, 0.25, 0.15, 0.07, 0.03)]
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

    tra = {}
    for r_, v_, f_ in zip(master.row_id.values, master.video_id.values,
                          master.frame_idx.values):
        tra.setdefault((v_, int(f_)), set()).add(int(r_))

    nen, v_dung = {}, {}
    for c in giu:
        cac = []
        for sk in R.tach_su_kien(c.cau_hoi):
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(sk)])
            cac.append(hop_nhat([anh, k3.tim(sk, k=a.be)], trong_so=[1.0, W3]))
        nen[c.id] = cac
        v_dung[c.id] = master.video_id.iloc[c.row_id_dung[0][0]]

    # ── tầng 1: video đúng nằm ở hạng mấy ────────────────────────────
    print("TẦNG 1 — CHỌN VIDEO (đo thẳng, không lẫn với khâu lắp chuỗi)\n")
    print(f"{'cách chấm':<26}{'hạng 1':>9}{'top-5':>8}{'top-20':>8}"
          f"{'ngoài bể':>10}")
    print("-" * 61)
    for ten, cham in CACH.items():
        h1 = t5 = t20 = ngoai = 0
        for c in giu:
            diem_v, _ = xep_video(nen[c.id], cham)
            xep = sorted(diem_v, key=lambda v: -diem_v[v])
            v = v_dung[c.id]
            if v not in diem_v:
                ngoai += 1
                continue
            h = xep.index(v) + 1
            h1 += h == 1
            t5 += h <= 5
            t20 += h <= 20
        n = len(giu)
        print(f"{ten:<26}{h1:>6}/{n}{t5:>6}/{n}{t20:>6}/{n}{ngoai:>8}/{n}")

    print("\n⚠️ `ngoài bể` = video đúng thiếu ứng viên cho ít nhất một sự kiện")
    print("   nên bị loại trước khi chấm. Cách chấm không cứu được nhóm này.")

    # ── tầng 2: điểm bài nộp ─────────────────────────────────────────
    for ds in DUNG_SAI:
        print(f"\n── điểm BÀI NỘP, dung sai ±{ds:g}s " + "─" * 30)
        print(f"{'cách chấm':<26}{'điểm':>9}")
        for ten, cham in CACH.items():
            tong = 0.0
            for c in giu:
                dung = [no_cua_so(b, master, ds) for b in c.row_id_dung]
                tong += m78.diem_bai_nop(
                    [[{r} for r in d] for d in lap(nen[c.id], master, cham)],
                    dung)
            print(f"{ten:<26}{tong / len(giu):>9.4f}")


if __name__ == "__main__":
    main()
