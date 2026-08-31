"""
56_do_mo_rong_truy_van.py — Đưa nhiều mệnh đề vào kênh 1 thế nào cho đúng?

    python scripts/56_do_mo_rong_truy_van.py --file dev/tap_de_that.jsonl

VẤN ĐỀ ĐO ĐƯỢC, KHÔNG PHẢI GIẢ THUYẾT

Tháp văn bản SigLIP2 có `context_length = 64`. Đo trên đề thật:

    cả câu nguyên : trung vị 64 token = ĐÚNG TRẦN, 12/20 câu bị cắt cụt
    từng mệnh đề  : trung vị 34 token, max 47

Nghĩa là **truyền cả câu vào là mất phần đuôi**, im lặng. `run.py` biết điều đó
nên gọi `tach_truy_van()` trước (A19/A20). Nhưng các phép đo A47–A50 lại truyền
cả câu — đo một cấu hình `run.py` không dùng. Script này sửa chỗ đó, và nhân
tiện thử luôn cách gộp mệnh đề tốt hơn.

BỐN CÁCH ĐƯA MỆNH ĐỀ VÀO, ĐO CÙNG NHAU

  1. cả câu nguyên      — cái A47–A50 đã đo. Bị cắt ở token 64.
  2. mệnh đề, max cosine — cái `run.py` đang làm. `tim()` lấy điểm cao nhất
                           trên từng keyframe qua các mệnh đề.
  3. mệnh đề, RRF hạng   — MỚI. Mỗi mệnh đề là một truy vấn riêng, lấy top-100
                           riêng, rồi hợp nhất bằng RRF.
  4. (2) hoặc (3) + cả câu — thêm câu nguyên như một "mệnh đề" nữa.

VÌ SAO (3) CÓ THỂ HƠN (2)

Cùng lý do repo này hợp nhất kênh bằng RRF chứ không cộng điểm: **cosine của
hai mệnh đề khác nhau không so được với nhau**. Mệnh đề "một người phụ nữ" khớp
mờ với hàng nghìn khung ở cos ~0,30; mệnh đề "biển hiệu màu tím ghi BỆNH VIỆN"
khớp đúng một khung ở cos ~0,25. Lấy max cosine thì mệnh đề *dễ* nuốt mất mệnh
đề *đặc trưng*. Xếp theo hạng thì mỗi mệnh đề có tiếng nói ngang nhau.

CHI PHÍ: BẰNG KHÔNG

Mọi chuỗi đã có sẵn trong cache — `25_ma_hoa_truy_van.py` vốn mã hoá cả câu
nguyên LẪN từng mệnh đề. Không cần GPU, không cần dữ liệu mới.
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
from dense import KenhAnhCache, be_chung              # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def nho(f):
    """Nhớ theo `id` câu — mỗi cấu hình được gọi lại ở hai mức dung sai."""
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
    ap.add_argument("--moc", default=None, help="doi moc nen sang cau hinh khac")
    ap.add_argument("--trong-so-phu", type=float, default=0.75,
                    help="trọng số kênh 3 khi hợp nhất (mặc định A50)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(
        [c.cau_hoi] + R.tach_truy_van(c.cau_hoi))]
    if len(giu) < len(cau):
        print(f"⚠️ loại {len(cau) - len(giu)} câu thiếu chuỗi trong cache")
    mp = [len(R.tach_truy_van(c.cau_hoi)) for c in giu]
    print(f"{a.file.name}: {len(giu)} câu | mệnh đề trung vị "
          f"{sorted(mp)[len(mp) // 2]}, {sum(1 for x in mp if x == 1)} câu chỉ 1\n")

    ca_cau = nho(lambda c: k1.tim(c.cau_hoi, k=100))
    md_max = nho(lambda c: k1.tim(R.tach_truy_van(c.cau_hoi), k=100))
    md_rrf = nho(lambda c: hop_nhat(
        [k1.tim(m, k=100) for m in R.tach_truy_van(c.cau_hoi)]))
    md_rrf_ca = nho(lambda c: hop_nhat(
        [k1.tim(m, k=100) for m in R.tach_truy_van(c.cau_hoi)]
        + [k1.tim(c.cau_hoi, k=100)]))
    f3 = nho(lambda c: k3.tim(c.cau_hoi, k=100))

    w = a.trong_so_phu

    def voi_ocr(f):
        return lambda c: hop_nhat([f(c), f3(c)], trong_so=[1.0, w])

    cau_hinh = {
        "1. cả câu (A47-A50 đo cái này)": ca_cau,
        "2. mệnh đề, max cosine (run.py)": md_max,
        "3. mệnh đề, RRF hạng": md_rrf,
        "4. mệnh đề RRF + cả câu": md_rrf_ca,
        f"2 + OCR({w})": voi_ocr(md_max),
        f"3 + OCR({w})": voi_ocr(md_rrf),
        f"4 + OCR({w})": voi_ocr(md_rrf_ca),
    }
    if a.moc:
        if a.moc not in cau_hinh:
            raise SystemExit(f"--moc phai la mot trong: {list(cau_hinh)}")
        cau_hinh = {a.moc: cau_hinh[a.moc],
                    **{k: v for k, v in cau_hinh.items() if k != a.moc}}
    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
