"""
87_do_hang_so_k.py — Hằng số `k` của RRF: dò vùng DƯỚI 20 mà A48 chưa chạm.

    python scripts/87_do_hang_so_k.py

VÌ SAO DÒ LẠI MỘT THỨ A48 ĐÃ KẾT LUẬN "GẦN NHƯ TRƠ"

A48 dò k ∈ {20, 30, 60, 100, 120, 200} và thấy k = 100 với 120 cho **0-0-50** —
không một câu nào đổi thứ hạng. Nhưng phép dò đó có hai chỗ nay đã khác:

  * chạy ở trọng số kênh 3 **1:1**, còn hiện tại là **0,5** (A52);
  * mốc nền lúc ấy 0,3440, chưa có RRF hạng cho mệnh đề (A51);
  * và **dừng ở k = 20**, không đi thấp hơn.

A86 đo được điều chỉ thẳng vào vùng chưa dò: ứng viên top-1 của kênh 3 rơi vào
**hạng trung vị 64** sau hợp nhất, chỉ **10% vào nổi top-20**, và **74% là
ứng viên CHỈ kênh 3 tìm ra**. Đó là trần do công thức, tính ra được:

    w/(k+1) = 1/(k+h)  ->  h = (k+1)/w − k

    k = 60 -> h = 62      k = 20 -> h = 22      k = 10 -> h = 12
    k = 30 -> h = 32      k = 15 -> h = 17      k =  5 -> h =  7

`h` là hạng thấp nhất của kênh 1 mà một ứng viên CHỈ kênh 3 tìm ra có thể vượt.
Điểm BTC là trung bình R@{1,5,20,50,100}, nên chừng nào h > 20 thì kênh 3
**không thể chạm vào ba mốc đầu**, bất kể nó đúng tới đâu. A48 dừng ở k = 20
tức h = 22 — sát ngay bên ngoài, chưa bao giờ bước qua.

⚠️ `k` nhỏ là con dao hai lưỡi, và đó chính là thứ cần đo: nó cũng làm ứng viên
top-1 CHỈ kênh 1 tìm ra áp đảo mạnh hơn, tức thưởng cho sự tự tin của cả hai
kênh. Không có lý do tiên nghiệm để nó thắng.
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
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
CAC_K = (5, 10, 15, 20, 30, 60)
MOC = 60


def nho(f):
    cache = {}

    def g(c):
        if c.id not in cache:
            cache[c.id] = f(c)[:100]
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
    print(f"{len(giu)} câu | kênh 3 trọng số {a.w3:g}\n")
    print("Trần hạng theo công thức:  h = (k+1)/w − k")
    for k in CAC_K:
        print(f"  k = {k:>3}  ->  ứng viên CHỈ kênh 3 vượt được tới hạng "
              f"{(k + 1) / a.w3 - k:.0f} của kênh 1")
    print()

    # ⚠️ `k` của RRF MỆNH ĐỀ giữ nguyên 60 ở mọi dòng — chỉ đổi `k` của RRF
    # TẦNG KÊNH. Đổi cả hai là đổi hai thứ một lúc.
    anh = nho(lambda c: hop_nhat(
        [k1.tim(m, k=100) for m in R.tach_truy_van(c.cau_hoi)]))
    ocr = nho(lambda c: k3.tim(c.cau_hoi, k=100))

    def voi(k):
        return nho(lambda c: hop_nhat([anh(c), ocr(c)],
                                      k=k, trong_so=[1.0, a.w3]))

    # ⚠️ `bao_cao_do_nhay` lấy mốc nền là mục ĐẦU TIÊN của dict — tham số
    # `moc` của nó là danh sách MỨC DUNG SAI, không phải tên cấu hình.
    # Nên k = 60 (cấu hình đang chạy) phải đứng đầu.
    cau_hinh = {f"k = {MOC:<3g} ← MỐC": voi(MOC)}
    for k in CAC_K:
        if k != MOC:
            cau_hinh[f"k = {k:<3g}"] = voi(k)
    cau_hinh["chỉ kênh 1 (chẩn đoán)"] = anh

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
