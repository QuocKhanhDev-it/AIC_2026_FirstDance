"""
106_do_ket_hop_kenh3.py — Hai ứng viên 🟡 CÙNG nằm trên kênh 3: cộng lại có qua ngưỡng không?

    python scripts/106_do_ket_hop_kenh3.py --file dev/tap_de_that.jsonl

VÌ SAO ĐO PHÉP KẾT HỢP, VÀ VÌ SAO NÓ KHÔNG PHẠM LUẬT "CHỈ ĐỔI MỘT THỨ"

Sau khi A89 dọn nhãn nhiễm, quét lại toàn bộ kết luận cũ trên 52 câu sạch còn
đúng **hai** thứ dương mà chưa qua ngưỡng — và cả hai đều tác động lên kênh 3:

    A88  văn bản GỘP (ocr cũ + VietOCR)   +0,0144 / +0,0000   🟡  (ngưỡng 0,0151)
    A82  khuếch tán thời gian τ=2s        +0,0038 / +0,0038   🟡  (ngưỡng 0,0134)

Chúng sửa hai khâu KHÁC nhau của cùng một kênh: một cái đổi **văn bản đầu vào**,
một cái đổi **cách điểm lan ra khung lân cận**. Không cái nào bao cái nào, nên
về nguyên tắc hiệu có thể cộng.

"Chỉ đổi một thứ" là luật về **quy công**, không phải luật cấm đo cấu hình
phức tạp. Ở đây bảng có đủ bốn ô của lưới 2×2 (gốc / chỉ gộp / chỉ khuếch tán /
cả hai), nên nếu "cả hai" thắng thì vẫn đọc được phần nào do đâu — và nếu nó
KHÔNG bằng tổng hai phần thì đó chính là bằng chứng hai cơ chế giẫm lên nhau.

⚠️ DỰ ĐOÁN GHI TRƯỚC (ghi trước khi chạy, để không tự chấm điểm sau khi biết
kết quả): tôi cho rằng hiệu sẽ KHÔNG cộng đủ để qua ngưỡng. Lý do: cả hai đều
chỉ giúp ở những câu mà kênh 3 vốn đã tìm đúng vùng, mà A71 đo được kênh 3 chỉ
chồng 2,9% với kênh 1 — phần lớn câu, kênh 3 không có gì để khuếch tán.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))
sys.path.insert(0, str(GOC / "scripts"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan, doc_van_ban_khung        # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

_kt = __import__("95_do_khuech_tan_thoi_gian")
KhungTheoVideo, khuech_tan = _kt.KhungTheoVideo, _kt.khuech_tan

W3 = 0.5
TAU = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--tau", type=float, default=TAU)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    kv = KhungTheoVideo(master)
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3_cu = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    k3_gop = KenhVanBan.tu_bang_khung(
        master, doc_van_ban_khung(a.index), cot="text", ten="ocr_gop")

    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)} câu | kênh 3 trọng số {W3:g} | τ = {a.tau:g}s\n")

    nen = {}

    def anh(c):
        if c.id not in nen:
            nen[c.id] = hop_nhat([k1.tim(m, k=a.be)
                                  for m in R.tach_truy_van(c.cau_hoi)])
        return nen[c.id]

    def dung(k3, tau):
        """Một ô của lưới 2×2: nguồn văn bản × có/không khuếch tán."""
        nho = {}

        def g(c):
            if c.id not in nho:
                v = k3.tim(c.cau_hoi, k=a.be)
                if tau > 0:
                    v = khuech_tan(v, kv, tau, gioi_han=a.be)
                nho[c.id] = hop_nhat([anh(c), v], trong_so=[1.0, W3])[:100]
            return nho[c.id]
        return g

    cau_hinh = {
        "1. MỐC: ocr cũ, không khuếch tán": dung(k3_cu, 0.0),
        "2. + văn bản GỘP (A88)":           dung(k3_gop, 0.0),
        f"3. + khuếch tán τ={a.tau:g}s (A82)": dung(k3_cu, a.tau),
        "4. CẢ HAI":                        dung(k3_gop, a.tau),
    }
    print(bao_cao_do_nhay(giu, cau_hinh, master))
    print("\nĐỌC BẢNG: dòng 4 thắng mà KHÔNG xấp xỉ tổng hiệu của dòng 2 và 3\n"
          "nghĩa là hai cơ chế giẫm lên nhau — lúc đó đừng cộng chúng vào nhau\n"
          "trong đầu ở những phép đo sau.")


if __name__ == "__main__":
    main()
