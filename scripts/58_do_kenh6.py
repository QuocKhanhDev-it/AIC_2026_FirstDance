"""
58_do_kenh6.py — Kênh 6 (OCR/ASR nhúng vector) có thêm được gì không?

    python scripts/58_do_kenh6.py
    python scripts/58_do_kenh6.py --file dev/tap_dev.jsonl --theo-nguon

CÂU HỎI

Kênh 3 dùng BM25 — khớp MẶT CHỮ. Truy vấn "xe cứu thương" mà bản tin viết "xe
cấp cứu" thì điểm bằng 0. Kênh 6 nhúng cùng văn bản đó vào không gian gopt nên
hai cách gọi nằm gần nhau. Đó là LÝ DO dựng nó, không phải bằng chứng nó chạy.

⚠️ CÓ MỘT LÝ DO CHÍNH ĐÁNG ĐỂ NGỜ

SigLIP2 huấn luyện để khớp **ảnh ↔ chữ**, không phải **chữ ↔ chữ**. Đem vector
truy vấn so với vector tài liệu là dùng model NGOÀI phân bố huấn luyện. Vì vậy
bảng dưới có dòng "chỉ kênh 6" — nếu nó gần 0 thì kênh này không đọc được gì,
và mọi con số hợp nhất phía trên chỉ là kênh 1 đội lốt.

BA CÂU HỎI KHÁC NHAU, ĐỪNG TRỘN

  * kênh 6 THÊM vào cấu hình đang chạy có lãi không?
  * kênh 6 THAY được kênh 3 không? (dense thay sparse)
  * kênh 6 đứng một mình có truy hồi được gì không? (chẩn đoán)

Mốc nền là cấu hình `run.py` ĐANG chạy sau A52: mệnh đề RRF hạng + kênh 3
trọng số 0,5. Chỉ đổi MỘT thứ so với mốc ở mỗi dòng.
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
from van_ban_dense import KenhVanBanDense             # noqa: E402

W3 = 0.5                                              # trọng số kênh 3 (A52)
W6 = (0.25, 0.5, 1.0)                                 # trọng số kênh 6 đem dò


def _id(f: Path) -> set:
    return {json.loads(l)["id"]
            for l in f.read_text("utf-8").splitlines() if l.strip()}


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
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--van-ban", default=GOC / "index" / "van_ban_gopt", type=Path,
                    help=".npz hoặc thư mục đã bung — cả hai đều đọc được")
    ap.add_argument("--cache", default=None, type=Path,
                    help="cache truy vấn CỦA KÊNH 6. Mặc định dùng chung cache "
                         "gopt — chỉ đúng khi kênh 6 nhúng bằng chính tháp văn "
                         "bản gopt. Model khác (BGE-M3…) thì BẮT BUỘC truyền "
                         "cache riêng: khác model là khác không gian vector, "
                         "dùng nhầm là hỏng câm.")
    ap.add_argument("--theo-nguon", action="store_true")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    k6 = KenhVanBanDense(str(a.index), a.van_ban,
                         str(a.cache or a.index / "truy_van_gopt.npz"))
    print(f"kênh 6: {k6.vec.shape[0]:,} đoạn / {len(k6._r_duy):,} keyframe "
          f"| {k6.ghi_chu.get('model')} | cache "
          f"{Path(a.cache).name if a.cache else 'truy_van_gopt.npz'}")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    if len(giu) < len(cau):
        print(f"⚠️ loại {len(cau) - len(giu)} câu thiếu chuỗi trong cache")
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu\n")

    anh = nho(lambda c: hop_nhat(
        [k1.tim(m, k=100) for m in R.tach_truy_van(c.cau_hoi)]))
    ocr = nho(lambda c: k3.tim(c.cau_hoi, k=100))
    # Kênh 6 nhận danh sách mệnh đề: `_vec_truy_van` tự trung bình rồi chuẩn
    # hoá, đúng cách `dense.KenhAnh` làm — hai kênh cùng một vector truy vấn.
    dense6 = nho(lambda c: k6.tim(R.tach_truy_van(c.cau_hoi), k=100))

    cau_hinh = {
        f"1. mốc: ảnh + kênh 3 ({W3:g})":
            lambda c: hop_nhat([anh(c), ocr(c)], trong_so=[1.0, W3]),
    }
    for w in W6:
        cau_hinh[f"2. + kênh 6 ({w:g})"] = (lambda w: lambda c: hop_nhat(
            [anh(c), ocr(c), dense6(c)], trong_so=[1.0, W3, w]))(w)
    cau_hinh[f"3. kênh 6 THAY kênh 3 ({W3:g})"] = lambda c: hop_nhat(
        [anh(c), dense6(c)], trong_so=[1.0, W3])
    cau_hinh["4. chỉ kênh 6 (chẩn đoán)"] = dense6
    cau_hinh["5. chỉ kênh 1 (chẩn đoán)"] = anh

    if not a.theo_nguon:
        print(bao_cao_do_nhay(giu, cau_hinh, master))
        return

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
