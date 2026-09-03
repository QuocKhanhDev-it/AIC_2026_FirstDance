"""
50_do_gopt.py — Ma trận thứ hai (gopt) có đáng bật không?

    python scripts/50_do_gopt.py --file dev/tap_de_that.jsonl

CÂU HỎI ĐANG ĐO, VIẾT CHO HẸP

`clip_gopt.npy` (ViT-gopt-16-SigLIP2-384, 1536 chiều) vừa phủ đủ 177.321 dòng.
Nó là **kênh 1 thứ hai**, độc lập với `clip_siglip2.npy` (SO400M, 1152 chiều).
Ba câu, theo đúng thứ tự:

  1. Một mình gopt có hơn một mình SigLIP2 không?
  2. RRF(gopt, SigLIP2) có hơn kênh 1 đơn mạnh nhất không?
  3. Thêm gopt vào cấu hình đang chạy — RRF(SigLIP2, OCR) — có lãi không?

Câu 3 mới là câu quyết định bật hay không. Câu 1 và 2 chỉ để hiểu vì sao.

MỐC NỀN LÀ CẤU HÌNH MẠNH NHẤT HIỆN CÓ, KHÔNG PHẢI CÁI TIỆN TAY

Mốc là **RRF(SigLIP2, OCR) trọng số 1:1** — đúng cấu hình `src/run.py` đang bật
mặc định sau A45. So với kênh 1 trần thì gần như chắc thắng, mà thắng vậy không
nói lên điều gì.

CHỈ ĐO TRÊN CÂU MÀ CẢ HAI CACHE ĐỀU CÓ

Hai ma trận dùng hai cache khác nhau (1152 và 1536 chiều). Câu nào thiếu ở một
bên thì LOẠI KHỎI CẢ HAI — nếu không, hai cấu hình chạy trên hai bộ câu khác
nhau và mọi so sánh đều vô nghĩa.

⚠️ Đọc `bao_cao_do_nhay`, đừng đọc điểm trung bình. Đảo dấu giữa hai mức dung
sai nghĩa là KHÔNG kết luận được, không phải "hơi hơn".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache, be_chung              # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def loc_du_cache(cau, *kenh):
    """Giữ câu mà MỌI kênh đều mã hoá được. Trả về (giữ, bỏ).

    Kiểm bằng chính chuỗi sẽ được tra (`c.cau_hoi` nguyên văn), không phải
    chuỗi đã tách — hai thứ đó khác nhau, và tra nhầm thì phép lọc này báo
    "đủ" trong khi lúc chạy thật vẫn ném `KeyError`.
    """
    giu, bo = [], []
    for c in cau:
        if all(not k.co_du([c.cau_hoi]) for k in kenh):
            giu.append(c)
        else:
            bo.append(c)
    return giu, bo


def nho_kenh(k, k_top=100, **kw):
    """Bọc một kênh thành `f(cau_hoi) -> list[Candidate]`, có nhớ kết quả.

    Truyền THẲNG `c.cau_hoi`, không tự tách — đúng quy ước của
    `26_do_rrf_siglip2_ocr.py`, để con số so được với A45.

    Nhớ lại là bắt buộc chứ không phải tối ưu vặt: `bao_cao_do_nhay` chấm ở
    HAI mức dung sai, và sáu cấu hình dùng chung ba kênh — không nhớ thì mỗi
    kênh chạy lại 12 lần trên cùng một câu.
    """
    cache = {}

    def f(c):
        if c.id not in cache:
            cache[c.id] = k.tim(c.cau_hoi, k=k_top, **kw)
        return cache[c.id]
    return f


def gop(*ds, trong_so=None):
    """RRF nhiều kênh. `hop_nhat` nhận MỘT danh sách các danh sách, và tham số
    `k` của nó là **hằng số RRF** chứ không phải số ứng viên — để mặc định."""
    return hop_nhat(list(ds), trong_so=trong_so)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--moc", default=None, metavar="TEN",
                    help="đổi mốc nền sang cấu hình khác (mặc định: RRF(SigLIP2,OCR))")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)
    print(f"tập đo: {a.file.name} — {len(cau)} câu\n")

    print("nạp kênh…")
    k_si = KenhAnhCache(str(a.index), str(a.index / "truy_van.npz"),
                        matrix="clip_siglip2.npy")
    k_go = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                        matrix="clip_gopt.npy")
    print(f"  SigLIP2 {k_si.mat.shape}  gopt {k_go.mat.shape}")

    # Bể ứng viên chung — BẮT BUỘC khi hai ma trận có thể lệch độ phủ.
    # Giờ cả hai đủ 177.321, nhưng vẫn khoá để phép đo không phụ thuộc vào
    # điều đó còn đúng hay không sau này.
    be = be_chung(k_si, k_go)
    print(f"  bể chung: {int(be.sum()):,} / {len(master):,} dòng")

    ocr = a.index / "ocr_asr.parquet"
    # cột văn bản của `ocr_asr.parquet` tên là `text`, không phải `caption`
    k_ocr = KenhVanBan.tu_bang_khung(master, pd.read_parquet(ocr),
                                     cot="text", ten="ocr_asr")
    print("  kênh 3 (OCR+ASR) sẵn sàng")

    giu, bo = loc_du_cache(cau, k_si, k_go)
    if bo:
        print(f"\n⚠️ loại {len(bo)} câu thiếu trong một cache: "
              f"{', '.join(c.id for c in bo)}")
    print(f"đo trên {len(giu)} câu\n")

    f_si = nho_kenh(k_si, be=be)
    f_go = nho_kenh(k_go, be=be)
    f_ocr = nho_kenh(k_ocr)

    cau_hinh = {
        "RRF(SigLIP2,OCR) MỐC": lambda c: gop(f_si(c), f_ocr(c)),
        "SigLIP2 một mình": f_si,
        "gopt một mình": f_go,
        "RRF(gopt,SigLIP2)": lambda c: gop(f_go(c), f_si(c)),
        "RRF(gopt,SigLIP2,OCR)": lambda c: gop(f_go(c), f_si(c), f_ocr(c)),
        "RRF(gopt,OCR)": lambda c: gop(f_go(c), f_ocr(c)),
    }
    if a.moc:
        # Đổi mốc nền: `bao_cao_do_nhay` lấy cấu hình ĐẦU TIÊN làm mốc, nên
        # muốn hỏi "bỏ SigLIP2 đi có hơn không" thì phải đưa cấu hình mới lên
        # đầu — so với mốc cũ không trả lời được câu đó.
        if a.moc not in cau_hinh:
            raise SystemExit(f"--moc phải là một trong: {list(cau_hinh)}")
        cau_hinh = {a.moc: cau_hinh[a.moc],
                    **{k: v for k, v in cau_hinh.items() if k != a.moc}}
    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
