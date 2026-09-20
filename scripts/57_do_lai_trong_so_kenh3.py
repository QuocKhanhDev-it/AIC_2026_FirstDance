"""
57_do_lai_trong_so_kenh3.py — Dò lại trọng số kênh 3 dưới cách đưa truy vấn ĐÚNG.

    python scripts/57_do_lai_trong_so_kenh3.py
    python scripts/57_do_lai_trong_so_kenh3.py --file dev/tap_dev.jsonl   # đối chứng

VÌ SAO PHẢI ĐO LẠI

A50 chọn `--trong-so-phu 0.75` khi script đo còn truyền CẢ CÂU vào kênh 1 —
tức kênh 1 đang chạy với truy vấn bị cắt ở token 64 (A51). Trọng số là một
con số nói "kênh 3 đáng tin bao nhiêu SO VỚI kênh 1". Làm kênh 1 mạnh lên thì
tỉ lệ đó đổi nền: 0,75 không còn cơ sở nào cả.

Mốc nền ở đây là cấu hình `run.py` ĐANG chạy sau A51 — mệnh đề RRF hạng, kênh 3
trọng số 0,75 — chứ không phải cấu hình yếu hơn nào tiện tay.

CHỈ ĐỔI MỘT THỨ: trọng số. Cách gộp mệnh đề giữ nguyên RRF hạng ở mọi dòng.
Hằng số k của RRF giữ 60 (muốn dò k thì dò riêng, xem 53_).

`w = 0` (bỏ hẳn kênh 3) có mặt trong bảng vì nó là câu hỏi thật: nếu không
trọng số nào thắng nó thì kênh 3 không đáng bật.
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

TRONG_SO = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
MOC = 0.75                                            # mặc định A50, cần soát lại


def _id(f: Path) -> set:
    return {json.loads(l)["id"]
            for l in f.read_text("utf-8").splitlines() if l.strip()}


def nho(f):
    """Nhớ theo `id` câu — mỗi cấu hình bị gọi lại ở hai mức dung sai."""
    cache = {}

    def g(c):
        if c.id not in cache:
            cache[c.id] = f(c)
        return cache[c.id]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--moc", type=float, default=MOC,
                    help="trọng số dùng làm mốc nền để so cặp")
    ap.add_argument("--theo-nguon", action="store_true",
                    help="tách báo cáo theo nguồn câu hỏi (xem 54_)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    if len(giu) < len(cau):
        print(f"⚠️ loại {len(cau) - len(giu)} câu thiếu chuỗi trong cache")
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu\n")

    # Kênh 1 giống hệt `run.py` sau A51: mỗi mệnh đề một truy vấn, gộp bằng RRF.
    anh = nho(lambda c: hop_nhat(
        [k1.tim(m, k=100) for m in R.tach_truy_van(c.cau_hoi)]))
    ocr = nho(lambda c: k3.tim(c.cau_hoi, k=100))

    def voi(w):
        if w == 0:
            return anh
        return lambda c: hop_nhat([anh(c), ocr(c)], trong_so=[1.0, w])

    ten = {w: (f"kênh 3 trọng số {w:<4g}" if w else "BỎ HẲN kênh 3 (w = 0)")
           for w in TRONG_SO}
    if a.moc not in ten:
        ten[a.moc] = f"kênh 3 trọng số {a.moc:<4g}"
    ten[a.moc] += "  ← MỐC"

    cau_hinh = {ten[a.moc]: voi(a.moc)}
    cau_hinh.update({ten[w]: voi(w) for w in TRONG_SO if w != a.moc})
    if not a.theo_nguon:
        print(bao_cao_do_nhay(giu, cau_hinh, master))
        return

    # Cùng bộ nhớ đệm cho cả ba nhóm — chỉ báo cáo là tách, phép truy hồi thì
    # không chạy lại. Chia y hệt 54_ để hai phép đo so được với nhau.
    that = _id(GOC / "dev" / "tap_de_that.jsonl")
    moi = _id(GOC / "dev" / "tap_dev_bonus-30-8.jsonl")

    def nguon(c):
        return ("đề thật" if c.id in that
                else "mới sát đề thật" if c.id in moi else "tự soạn cũ")

    for ten in ("đề thật", "mới sát đề thật", "tự soạn cũ"):
        nhom = [c for c in giu if nguon(c) == ten]
        print()
        print("=" * 74)
        print(f"NHÓM: {ten}  ({len(nhom)} câu)")
        print("=" * 74)
        print(bao_cao_do_nhay(nhom, cau_hinh, master))


if __name__ == "__main__":
    main()
