"""
100_do_gopt_vs_siglip2.py — Hệ thống hiện tại đạt tới đâu, và hơn bản cũ bao nhiêu?

    python scripts/100_do_gopt_vs_siglip2.py

Trả lời hai câu trong MỘT lượt chạy:

  1. **Điểm cao nhất hệ thống hiện tại có thể đạt** — tức TRẦN: đáp án nằm đâu
     đó trong bể thì coi như xếp lại hoàn hảo cho 1,0. Khoảng cách giữa trần và
     điểm thật là dư địa của mọi phép xếp lại (A54).
  2. **Hơn bản chỉ dùng SigLIP2-1152 bao nhiêu** — `ViT-SO400M-14-SigLIP2-378`,
     1152 chiều, ma trận `clip_siglip2.npy` mà A17 từng đưa kênh 1 từ 0,0000
     lên 0,3258.

HAI CHỖ PHẢI KHOÁ, KHÔNG THÌ SỐ ĐO VÔ NGHĨA

**1. Khoá TẬP CÂU.** Cache truy vấn của hai model phủ khác nhau — gopt có 1.468
chuỗi (đủ 72 câu), cache cũ chỉ 1.158 (đủ 52 câu). Đo gopt trên 72 câu rồi so
với SigLIP2 trên 52 câu là so hai bộ đề khác nhau. Script tự lọc về phần **cả
hai cùng đủ chuỗi**.

**2. KHÔNG cần khoá bể ứng viên.** Cả hai ma trận đều phủ trọn 177.321 dòng
(đã kiểm kích thước file), nên bẫy `dense.be_chung` ở A17 — bể nhỏ hơn thắng vì
lý do không liên quan tới chất lượng, đo được +0,2833 — không áp dụng ở đây.

⚠️ ĐÍNH CHÍNH A54 ĐƯỢC TÔN TRỌNG: mọi cấu hình đo trong CÙNG một lượt chạy,
trên CÙNG bộ câu. A54 từng đọc chênh lệch giữa ba lượt chạy riêng như một hiệu
ứng thật, và phải tự đính chính.
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import (DUNG_SAI_CHINH, DUNG_SAI_KIEM,  # noqa: E402
                       bao_cao_do_nhay, no_cua_so)
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
MA_TRAN = {
    "gopt-1536": ("clip_gopt.npy", "truy_van_gopt.npz"),
    "siglip2-1152": ("clip_siglip2.npy", "truy_van.npz"),
}


def du_chuoi(c, kenh) -> bool:
    if c.loai == "TRAKE":
        return not any(kenh.co_du(R.tach_truy_van(sk))
                       for sk in R.tach_su_kien(c.cau_hoi))
    return not kenh.co_du(R.tach_truy_van(c.cau_hoi))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--be", type=int, default=1000,
                    help="cỡ bể để tính TRẦN. Bài nộp vẫn 100 dòng")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)

    # ── khoá tập câu: chỉ giữ câu CẢ HAI cache cùng đủ chuỗi ─────────
    du = {}
    for ten, (mat, cache) in MA_TRAN.items():
        k1 = KenhAnhCache(str(a.index), str(a.index / cache), matrix=mat)
        du[ten] = {c.id for c in cau if du_chuoi(c, k1)}
        print(f"  {ten:<14}{len(du[ten])}/{len(cau)} câu đủ chuỗi")
        del k1
        gc.collect()
    giu = [c for c in cau if all(c.id in s for s in du.values())]
    print(f"\nTẬP KHOÁ: {len(giu)}/{len(cau)} câu cả hai cùng đo được\n")

    # ── tính sẵn ứng viên cho từng ma trận, rồi THẢ ma trận ──────────
    anh = {}
    for ten, (mat, cache) in MA_TRAN.items():
        k1 = KenhAnhCache(str(a.index), str(a.index / cache), matrix=mat)
        print(f"  quét {ten}: {k1.mat.shape}")
        anh[ten] = {c.id: hop_nhat([k1.tim(m, k=a.be)
                                    for m in R.tach_truy_van(c.cau_hoi)])[:a.be]
                    for c in giu}
        del k1
        gc.collect()
    ocr = {c.id: k3.tim(c.cau_hoi, k=a.be) for c in giu}
    print()

    def hop(ten):
        return lambda c: hop_nhat([anh[ten][c.id], ocr[c.id]],
                                  trong_so=[1.0, W3])[:100]

    cau_hinh = {
        "gopt + kênh 3  ← ĐANG CHẠY": hop("gopt-1536"),
        "gopt một mình": lambda c: anh["gopt-1536"][c.id][:100],
        "SigLIP2-1152 + kênh 3": hop("siglip2-1152"),
        "SigLIP2-1152 một mình": lambda c: anh["siglip2-1152"][c.id][:100],
    }
    print(bao_cao_do_nhay(giu, cau_hinh, master))

    # ── TRẦN: đáp án nằm đâu đó trong bể -> xếp lại hoàn hảo cho 1,0 ──
    print(f"\n\nTRẦN — đáp án có nằm trong bể {a.be} không "
          f"(xếp lại hoàn hảo thì được 1,0)\n")
    print(f"{'cấu hình':<30}" + "".join(f"{'±' + str(d) + 's':>10}"
                                        for d in (DUNG_SAI_CHINH, DUNG_SAI_KIEM)))
    print("-" * 50)
    for ten, f in (("gopt + kênh 3", hop("gopt-1536")),
                   ("SigLIP2-1152 + kênh 3", hop("siglip2-1152"))):
        # dùng bể ĐẦY ĐỦ chứ không cắt 100 — trần là của bể, không của bài nộp
        be = {c.id: hop_nhat([anh[ten.split()[0].replace("gopt", "gopt-1536")
                                  .replace("SigLIP2-1152", "siglip2-1152")][c.id],
                              ocr[c.id]], trong_so=[1.0, W3])[:a.be]
              for c in giu}
        o = []
        for ds in (DUNG_SAI_CHINH, DUNG_SAI_KIEM):
            n = 0
            for c in giu:
                if c.loai == "TRAKE":
                    continue
                dung = no_cua_so(c.row_id_dung, master, ds)
                n += any(x.row_id in dung for x in be[c.id])
            o.append(n / sum(1 for c in giu if c.loai != "TRAKE"))
        print(f"{ten:<30}" + "".join(f"{x:>10.4f}" for x in o))
    print(f"\n(TRẦN tính trên {sum(1 for c in giu if c.loai != 'TRAKE')} câu "
          f"KIS/QA — TRAKE chấm theo vị trí nên 'có trong bể' không cùng nghĩa)")


if __name__ == "__main__":
    main()
