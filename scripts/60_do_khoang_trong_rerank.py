"""
60_do_khoang_trong_rerank.py — Xếp lại bể ứng viên còn moi được bao nhiêu điểm?

    python scripts/60_do_khoang_trong_rerank.py
    python scripts/60_do_khoang_trong_rerank.py --be 100 300 1000

CÂU HỎI

Reranker (VLM chấm lại từng ứng viên) là món đắt: mỗi câu vài chục tới vài trăm
lượt gọi model ảnh–chữ. Trước khi bỏ công, phải biết **nó có chỗ để thắng
không**. Chỗ đó đo được chính xác, không cần đoán:

    TRẦN   = đáp án nằm ĐÂU ĐÓ trong bể ứng viên  ->  xếp lại hoàn hảo cho 1,0
    THẬT   = điểm hiện tại (trung bình R@{1,5,20,50,100} theo công thức BTC)
    TRỐNG  = TRẦN − THẬT  ->  đúng phần một reranker HOÀN HẢO có thể lấy

Đây là trần CỨNG cho mọi hậu xử lý: xếp lại, mở rộng lân cận, tác tử VLM đều
chỉ hoán vị hoặc mở rộng quanh bể có sẵn. Đáp án không nằm trong bể thì không
cách nào cứu — phải sửa TRUY HỒI, không phải sửa xếp hạng.

VÌ SAO ĐO NHIỀU CỠ BỂ

Bài nộp chỉ được 100 dòng, nhưng **bể ứng viên không bị giới hạn đó**. Một
reranker có thể xét top-1000 rồi nộp 100 dòng tốt nhất. Nếu trần ở 1000 cao
hơn hẳn ở 100 thì chỉ riêng việc nới bể đã là một khoản lãi — và nó rẻ hơn
reranker nhiều.

⚠️ TRAKE chấm từng phần theo sự kiện nên không có "hạng" duy nhất; bảng dưới
tách riêng, và cột trần của TRAKE là điểm khi MỌI sự kiện tìm được đều được
kéo lên hạng 1.
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
from cham_diem import (MOC, cham, diem_cau, r_at_k)   # noqa: E402
from cham_diem import DUNG_SAI_CHINH, DUNG_SAI_KIEM   # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5                                              # trọng số kênh 3 (A52)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, nargs="+", default=[100, 300, 1000],
                    help="các cỡ bể ứng viên đem đo")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    n_trake = sum(1 for c in giu if c.loai == "TRAKE")
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu "
          f"({len(giu) - n_trake} KIS/QA + {n_trake} TRAKE)\n")

    for be in a.be:
        # Cấu hình y hệt `run.py` sau A52, chỉ nới cỡ bể.
        def chay(c, be=be):
            anh = hop_nhat([k1.tim(m, k=be) for m in R.tach_truy_van(c.cau_hoi)])
            return hop_nhat([anh, k3.tim(c.cau_hoi, k=be)],
                            trong_so=[1.0, W3])[:be]

        print("=" * 78)
        print(f"BỂ ỨNG VIÊN {be} dòng")
        print("=" * 78)
        for ds in (DUNG_SAI_CHINH, DUNG_SAI_KIEM):
            b = cham(giu, chay, gioi_han=be, master=master, dung_sai_giay=ds)
            kis = b[b.loai != "TRAKE"]
            tr = b[b.loai == "TRAKE"]

            # THẬT: điểm BTC hiện tại. TRẦN: đáp án lọt bể -> kéo lên hạng 1.
            that = b.diem.mean()
            lot = kis.hang.notna()
            tran_kis = float(lot.mean())              # mỗi câu lọt bể -> 1,0
            # TRAKE: `diem` đã là tỷ lệ sự kiện khớp trong bể; kéo lên hạng 1
            # thì mỗi sự kiện TÌM ĐƯỢC ăn trọn điểm, tức tỷ lệ sự kiện có mặt.
            tran_trake = (tr.diem.apply(lambda d: 1.0 if d > 0 else 0.0).mean()
                          if len(tr) else float("nan"))
            tran = (tran_kis * len(kis) + (tran_trake * len(tr) if len(tr) else 0)
                    ) / len(b)

            bac = {k: r_at_k_tb(kis.hang, k) for k in MOC}
            print(f"\n  dung sai ±{ds:g}s")
            print(f"    {'R@' + str(MOC[0]):>7}"
                  + "".join(f"{'R@' + str(k):>8}" for k in MOC[1:]))
            print(f"    {bac[MOC[0]]:>7.4f}"
                  + "".join(f"{bac[k]:>8.4f}" for k in MOC[1:]))
            print(f"    điểm THẬT (BTC)          {that:.4f}")
            print(f"    TRẦN nếu xếp lại hoàn hảo {tran:.4f}"
                  f"   (KIS/QA {tran_kis:.4f}"
                  + (f", TRAKE {tran_trake:.4f}" if len(tr) else "") + ")")
            print(f"    KHOẢNG TRỐNG              {tran - that:+.4f}"
                  f"   = {(tran - that) * 100:.1f} điểm phần trăm")
            if len(kis):
                ngoai = int((~lot).sum())
                print(f"    KIS/QA đáp án NGOÀI bể: {ngoai}/{len(kis)} câu "
                      f"— truy hồi phải sửa, xếp lại vô ích")
        print()


def r_at_k_tb(hang: pd.Series, k: int) -> float:
    return float(hang.apply(lambda h: r_at_k(None if pd.isna(h) else int(h), k)).mean())


if __name__ == "__main__":
    main()
