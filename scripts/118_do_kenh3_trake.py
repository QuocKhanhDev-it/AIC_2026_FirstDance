"""
118_do_kenh3_trake.py — TRAKE có nên dùng kênh 3 không? (bài nộp hiện KHÔNG dùng)

    python scripts/118_do_kenh3_trake.py

LỖI PHÁT HIỆN ĐƯỢC KHI CHUẨN BỊ NỘP

`run.quet_van_ban()` chỉ nạp ứng viên kênh 2/3 cho câu **không phải TRAKE**:

    for ten, nd in de.items():
        if loai_cua(ten) != "trake":          # <- TRAKE bị loại ở đây
            ra.setdefault(ten, []).append(k3.tim(...))

và `phu[ten]` chỉ được hợp nhất trong nhánh `else` (không phải TRAKE). Nên
**bài nộp chạy TRAKE bằng kênh 1 một mình**.

Nhưng **mọi script đo TRAKE** — `78_` (A79 K-best), `89_` (A78 chấm video),
`91_` (A86 ngân sách), `92_` (A86 lại ghép), `110_`/`111_` (A94 bể) — đều dựng
ứng viên bằng:

    hop_nhat([anh, k3.tim(sk, k=be)], trong_so=[1.0, 0.5])

Tức **toàn bộ dòng kết luận TRAKE của repo đo trên một cấu hình bài nộp không
chạy.** Đây đúng loại hỏng đã sinh ra bốn lỗi im lặng ở commit `8a27e29`:
*"script đo không chạy cùng đường với bài nộp"*.

LƯỚI 2×2 — SỬA CÁI NÀO THÌ PHẢI BIẾT CÁI KIA

    kênh 1 một mình,  bể 100     <- BÀI NỘP đang chạy, chưa từng được đo
    kênh 1 + kênh 3,  bể 100     <- MỌI phép đo TRAKE giả định
    kênh 1 một mình,  bể 300
    kênh 1 + kênh 3,  bể 300     <- A94 đề xuất

Bốn ô này trả lời cả hai câu cùng lúc: kênh 3 có đáng cho TRAKE không, và bể
lớn có còn thắng khi không có kênh 3 không. Nếu chỉ đo một chiều thì sửa xong
vẫn không biết mình vừa sửa cái gì.

⚠️ Chấm ở tầng NỘP (`cham_trake_nhieu_muc`), không phải tầng KÊNH.

⚠️ DỰ ĐOÁN GHI TRƯỚC: kênh 3 sẽ giúp, vì sự kiện TRAKE thường là cảnh có chữ
(bảng hiệu, biển báo) và A45 đo được kênh 3 giúp ở KIS/Q&A. Nhưng tôi KHÔNG
chắc bể 300 còn thắng khi bỏ kênh 3 — A94 quy hiệu ứng cho "giao của N tập
nhỏ đi theo cấp số nhân", mà bỏ một kênh thì mỗi tập nhỏ đi, tức bể càng cần
lớn. Nếu vậy hiệu của bể 300 sẽ TĂNG ở dòng kênh-1-một-mình.
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
from cham_diem import bao_cao_tu_bang                 # noqa: E402
from cham_diem import cham_trake_nhieu_muc            # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

import kbest_trake as KB                              # noqa: E402

W3 = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl",
        GOC / "dev" / "tap_dev_trake.jsonl"])
    ap.add_argument("--be", type=int, nargs="*", default=[100, 300])
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
    print(f"\n{len(giu)} câu TRAKE\n")

    nho = {}

    def ung_vien(c, be, co_k3):
        khoa = (c.id, be, co_k3)
        if khoa not in nho:
            ds = []
            for sk in R.tach_su_kien(c.cau_hoi):
                anh = hop_nhat([k1.tim(m, k=be) for m in R.tach_truy_van(sk)])
                ds.append(hop_nhat([anh, k3.tim(sk, k=be)],
                                   trong_so=[1.0, W3])[:be] if co_k3
                          else anh[:be])
            nho[khoa] = ds
        return nho[khoa]

    def lam(be, co_k3):
        return lambda c: KB.lap_dong(ung_vien(c, be, co_k3), master, 100)

    # Chẩn đoán: kênh 3 đổi số video ứng viên bao nhiêu?
    import statistics as st
    print("CHẨN ĐOÁN — số video đủ ứng viên cho MỌI sự kiện (trung vị)")
    print(f"  {'bể':>6}{'kênh 1':>12}{'kênh 1+3':>12}")
    print("  " + "-" * 30)
    for be in a.be:
        m = []
        for co in (False, True):
            m.append(int(st.median(
                [len(KB.cham_video(KB.gom_theo_video(ung_vien(c, be, co))))
                 for c in giu])))
        print(f"  {be:>6}{m[0]:>12}{m[1]:>12}")
    print()

    ten = {}
    for be in a.be:
        for co in (False, True):
            nhan = (f"{'MỐC: ' if (be == a.be[0] and not co) else ''}"
                    f"{'kênh 1+3' if co else 'kênh 1'}, bể {be}")
            ten[nhan] = lam(be, co)
    bang = {k: cham_trake_nhieu_muc(giu, f, master) for k, f in ten.items()}
    print(bao_cao_tu_bang(bang))
    print("\nĐỌC BẢNG: mốc nền là cấu hình BÀI NỘP đang chạy (kênh 1, bể 100)"
          "\n— thứ chưa từng được đo. Ba dòng còn lại đều là thay đổi cần cân"
          "\nnhắc, và dòng `kênh 1+3, bể 100` chính là thứ MỌI phép đo TRAKE"
          "\ncũ đã giả định là đang chạy.")


if __name__ == "__main__":
    main()
