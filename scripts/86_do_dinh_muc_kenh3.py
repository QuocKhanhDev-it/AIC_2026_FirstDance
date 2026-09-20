"""
86_do_dinh_muc_kenh3.py — Kênh 3 có bị TRẦN HẠNG do chính công thức RRF không?

    python scripts/86_do_dinh_muc_kenh3.py

PHÉP TÍNH DẪN TỚI PHÉP ĐO NÀY

RRF cộng `w / (k + hạng)`. Với `k = 60` và trọng số kênh 3 `w = 0,5`:

    ứng viên top-1 CHỈ kênh 3 tìm ra : 0,5 / 61  = 0,008197
    ứng viên hạng 62 của kênh 1      : 1,0 / 122 = 0,008197

Nghĩa là **một ứng viên chỉ do kênh 3 tìm ra, dù là top-1 của nó, không bao giờ
xếp trên hạng 62 của kênh 1**. Mà A71 đo được chồng@20 giữa hai kênh chỉ 3,4% —
tức PHẦN LỚN ứng viên tốt của kênh 3 rơi đúng vào tình huống "chỉ kênh 3".

Điểm BTC là trung bình R@{1, 5, 20, 50, 100}. Nếu trần hạng trên là thật thì
kênh 3 chỉ với tới được **R@100 và R@50** — 2/5 số mốc — bất kể nó đúng đến đâu.

HAI THỨ ĐO Ở ĐÂY

**1. Chẩn đoán (không đổi gì).** Top-1/3/5 của kênh 3 rơi vào hạng nào của danh
sách đã hợp nhất? Bao nhiêu phần trăm là "chỉ kênh 3"? Nếu trần hạng không có
thật thì mọi thứ dưới đây khỏi bàn.

**2. Định mức chỗ dành riêng.** Vì "dòng sai không bị phạt" và ngân sách là 100
DÒNG, có thể cắt M dòng cuối của danh sách hợp nhất và dành cho top-M của riêng
kênh 3. Không đụng vào công thức RRF, không đổi trọng số.

⚠️ Cảnh báo tự đặt trước khi chạy: định mức ở ĐUÔI chỉ có thể đổi R@100 (và
R@50 nếu M lớn). Nếu ứng viên tốt của kênh 3 vốn đã nằm trong top-100 rồi thì
định mức không đổi gì cả — và đó là kết quả hợp lệ, không phải phép đo hỏng.
"""

import argparse
import statistics
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
DINH_MUC = (10, 20, 30)


def dat_cho(hop: list[Candidate], k3: list[Candidate],
            m: int, tong: int = 100) -> list[Candidate]:
    """Giữ `tong - m` dòng đầu của danh sách hợp nhất, dành `m` dòng cho kênh 3.

    Ứng viên của kênh 3 đã có mặt trong phần giữ lại thì bỏ qua — dành chỗ cho
    cái tiếp theo, chứ không để trống.
    """
    giu = hop[:tong - m]
    da = {c.row_id for c in giu}
    them = []
    for c in k3:
        if c.row_id not in da:
            them.append(c)
            da.add(c.row_id)
            if len(them) >= m:
                break
    return giu + them


def nho(f):
    cache = {}

    def g(c):
        if c.id not in cache:
            cache[c.id] = f(c)
        return cache[c.id]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--w3", type=float, default=W3)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)} câu đo được | kênh 3 trọng số {a.w3:g}\n")

    anh = nho(lambda c: hop_nhat(
        [k1.tim(m, k=1000) for m in R.tach_truy_van(c.cau_hoi)])[:1000])
    ocr = nho(lambda c: k3.tim(c.cau_hoi, k=1000))
    gop = nho(lambda c: hop_nhat([anh(c), ocr(c)], trong_so=[1.0, a.w3]))

    # ── 1. chẩn đoán: top-N của kênh 3 rơi vào đâu ────────────────────
    print("CHẨN ĐOÁN — top-N của kênh 3 nằm ở hạng nào sau khi hợp nhất\n")
    print(f"{'top-N kênh 3':<14}{'hạng trung vị':>15}{'vào top-20':>12}"
          f"{'vào top-100':>13}{'chỉ kênh 3':>13}")
    print("-" * 67)
    for n in (1, 3, 5):
        hang, t20, t100, rieng, dem = [], 0, 0, 0, 0
        for c in giu:
            trong_anh = {u.row_id for u in anh(c)[:1000]}
            vt = {u.row_id: i for i, u in enumerate(gop(c), 1)}
            for u in ocr(c)[:n]:
                dem += 1
                h = vt.get(u.row_id)
                if h:
                    hang.append(h)
                    t20 += h <= 20
                    t100 += h <= 100
                rieng += u.row_id not in trong_anh
        tv = statistics.median(hang) if hang else float("nan")
        print(f"{'top-' + str(n):<14}{tv:>15.0f}{t20 / dem * 100:>11.0f}%"
              f"{t100 / dem * 100:>12.0f}%{rieng / dem * 100:>12.0f}%")

    print("\nĐỌC: nếu 'vào top-20' ~0% thì trần hạng là THẬT — không ứng viên "
          "nào\ndo riêng kênh 3 tìm ra chạm được R@1, R@5, R@20.\n")

    # ── 2. định mức chỗ dành riêng ────────────────────────────────────
    cau_hinh = {"1. MỐC: RRF thuần": nho(lambda c: gop(c)[:100])}
    for m in DINH_MUC:
        cau_hinh[f"2. dành {m} dòng cuối cho kênh 3"] = nho(
            (lambda m: lambda c: dat_cho(gop(c), ocr(c), m))(m))
    cau_hinh["3. chỉ kênh 1 (chẩn đoán)"] = nho(lambda c: anh(c)[:100])

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
