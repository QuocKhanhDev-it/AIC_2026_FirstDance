"""
93_do_phat_khoang_cach.py — TRAKE: phạt MỀM theo khoảng cách thời gian.

    python scripts/93_do_phat_khoang_cach.py

HƯỚNG TRAKE CUỐI CÙNG CHƯA ĐO — và nó đã đổi nghĩa một nửa

Đề xuất gốc: thay "ép tăng dần NGẶT + nội suy chỗ thiếu" bằng phạt mềm
`DP[i,t] = S[i,t] + max_τ(DP[i-1,τ] − λ(t−τ))`.

Nhưng A79 đã bật K-best, và K-best **vốn không nội suy** — beam sinh chuỗi tăng
dần thật, không chèn khung đoán vào chỗ thiếu. Nên nửa "bỏ nội suy" của đề xuất
đã có sẵn. Phần còn áp dụng được là **phạt theo độ dài khoảng cách** khi nối:

    điểm chuỗi = Σ điểm ứng viên − λ · Σ (khoảng cách giây tới sự kiện trước)

Ý: chuỗi có các sự kiện cách nhau đều và gần thì đáng tin hơn chuỗi nhảy cóc
qua nửa video.

CHỌN DẢI λ BẰNG ĐƠN VỊ, KHÔNG BẰNG CẢM TÍNH

Đề xuất gợi ý λ ∈ [0,001; 0,01], nhưng dải đó phụ thuộc thang điểm của hệ
thống họ. Ở đây điểm là RRF, nằm khoảng **0,008–0,03**, còn khoảng cách giữa
hai sự kiện là **hàng chục tới hàng trăm giây**. Để `λ · khoảng_cách` sánh được
với điểm ứng viên (~0,01) ở khoảng cách ~50 giây thì λ ~ **0,0002**.

Nên dò quanh đó: 0 (tắt) / 0,00005 / 0,0002 / 0,001 / 0,005. Mức cuối đủ mạnh
để khoảng cách áp đảo hoàn toàn điểm ứng viên — có ở đó để thấy hình dạng
đường cong, không phải vì kỳ vọng nó thắng.

⚠️ Mốc nền là cấu hình `run.py` VỪA BẬT ở A79 (K-best, giãn 3,0s, ngân sách
40/25/15/12/8, 20 dòng đuôi), tức mốc nền mạnh nhất hiện có.
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
from cham_diem import (bao_cao_tu_bang,               # noqa: E402
                       cham_trake_nhieu_muc)
from dense import KenhAnhCache                        # noqa: E402
from kbest_trake import lap_dong, phat_bac            # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
CAC_LAMDA = (0.0, 0.00005, 0.0002, 0.001, 0.005)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl",
        GOC / "dev" / "tap_dev_trake.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--lamda", type=float, nargs="*", default=list(CAC_LAMDA))
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

    nen = {}
    for c in giu:
        cac = []
        for sk in R.tach_su_kien(c.cau_hoi):
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(sk)])
            cac.append(hop_nhat([anh, k3.tim(sk, k=a.be)], trong_so=[1.0, W3]))
        nen[c.id] = cac

    # Khoảng cách thật giữa hai sự kiện liền kề — để biết dải λ có hợp lý không.
    pts = master.pts_time.values
    kc = [pts[b[0] if isinstance(b, list) else b] for c in giu
          for b in c.row_id_dung]
    hieu = []
    for c in giu:
        t = [pts[b[0] if isinstance(b, list) else b] for b in c.row_id_dung]
        hieu += [y - x for x, y in zip(t, t[1:])]
    hieu.sort()
    print(f"khoảng cách THẬT giữa hai sự kiện liền kề (n={len(hieu)}): "
          f"trung vị {hieu[len(hieu) // 2]:.1f}s, "
          f"min {hieu[0]:.1f}s, max {hieu[-1]:.1f}s")
    print(f"-> λ · trung vị = {0.0002 * hieu[len(hieu) // 2]:.5f} ở λ=0,0002, "
          f"sánh được với điểm RRF ~0,01\n")
    assert kc, "không có mốc thời gian nào"

    def nho(f):
        c_ = {}

        def g(c):
            if c.id not in c_:
                c_[c.id] = f(c)
            return c_[c.id]
        return g

    cau_hinh = {}
    for lam in a.lamda:
        ten = ("λ = 0 (TẮT) ← MỐC" if lam == 0 else f"tỷ lệ thuận λ={lam:g}")
        cau_hinh[ten] = nho(
            (lambda lam: lambda c: lap_dong(nen[c.id], master,
                                            phat_giay=lam))(lam))

    # Hàm phạt dạng BẬC — A80 bác phạt TỶ LỆ THUẬN, không bác cái này. Vùng
    # [gần, xa] không bị phạt, mà đó là nơi chứa phần lớn khoảng cách thật.
    for gan, nang, xa, beta in ((1.0, 1.0, 60.0, 0.0005),
                                (1.0, 1.0, 60.0, 0.002),
                                (1.5, 1.0, 60.0, 0.0005),
                                (1.0, 0.02, 60.0, 0.0002),
                                (1.0, 1.0, 30.0, 0.0005)):
        ten = f"bậc <{gan:g}s:{nang:g} >{xa:g}s:{beta:g}"
        cau_hinh[ten] = nho(
            (lambda g, n_, x, b: lambda c: lap_dong(
                nen[c.id], master, phat=phat_bac(g, n_, x, b)))(gan, nang, xa, beta))

    bang = {ten: cham_trake_nhieu_muc(giu, f, master)
            for ten, f in cau_hinh.items()}
    print(bao_cao_tu_bang(bang))


if __name__ == "__main__":
    main()
