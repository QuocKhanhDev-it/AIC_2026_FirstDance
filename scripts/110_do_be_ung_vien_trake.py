"""
110_do_be_ung_vien_trake.py — TRAKE thiếu VIDEO ứng viên, không thiếu cách xếp hạng.

    python scripts/110_do_be_ung_vien_trake.py

PHÁT HIỆN DẪN TỚI PHÉP ĐO NÀY (`109_do_xep_video_beam.py`)

Chẩn đoán hạn ngạch dòng của cấu hình đang chạy, trên 18 câu TRAKE:

    video có đủ ứng viên cho MỌI sự kiện : trung vị 11   (min 4, max 40)
    trong 25 video được chia dòng,
      số video KHÔNG có chuỗi hợp lệ     : trung vị 6    (min 0, max 25)
    số dòng thực sự nộp được             : trung vị 46/100  (min 0, max 81)

Hạn ngạch `40/25/15/12/8 + 20 dòng đuôi` được thiết kế cho **25 video**. Thực
tế trung vị chỉ có **11 video** đủ ứng viên, và một nửa trong số đó không tạo
nổi chuỗi tăng dần. Nên hơn một nửa số dòng bỏ trống — không phải vì khâu lắp
kém, mà vì **không có giả thuyết nào để lắp**.

VÌ SAO BỂ NHỎ, VÀ VÌ SAO ĐÓ LÀ NÚT THẮT THẬT

Mỗi sự kiện lấy `--be` ứng viên (mặc định **100**). Một video chỉ vào được danh
sách khi nó có ứng viên cho **TẤT CẢ** N sự kiện. Với N = 3 và 100 ứng viên mỗi
sự kiện trải trên nhiều video, giao của ba tập đó nhỏ đi rất nhanh — và nó nhỏ
đi theo cấp số nhân với N.

Đây KHÁC với `--be` của KIS/Q&A: ở đó 100 ứng viên là 100 dòng nộp, tăng lên
không giúp gì vì chỉ nộp được 100. Ở TRAKE, `--be` là **bể để GIAO**, và cái
nộp đi là chuỗi ghép từ giao đó. Cùng một tham số, hai vai trò hoàn toàn khác.

⚠️ VÌ SAO KHÔNG PHẢI THỨ A86 ĐÃ BÁC. A86 đo "bù dòng trống cho đủ 100" và thấy
vô ích — đúng, vì bù bằng dòng bừa thì dòng bừa được 0 điểm. Ở đây không bù
dòng: ta làm cho **có thêm video ứng viên thật** để sinh ra dòng thật.

⚠️ DỰ ĐOÁN GHI TRƯỚC: bể tăng thì số video tăng và số dòng nộp được tăng —
điều đó gần như chắc chắn. Nhưng điểm có tăng hay không thì KHÔNG chắc: video
thứ 12–25 là những video mà kênh xếp hạng thấp, và A86 đã cho thấy đuôi danh
sách hầu như không chứa đáp án. Tôi cho rằng ±15s sẽ tăng rõ hơn ±2s, vì dòng
thêm vào chủ yếu đúng video nhưng lệch thời điểm.

⚠️ Chấm ở tầng NỘP. Chỉ đổi MỘT thứ: `--be`.
"""

import argparse
import statistics as st
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
CAC_BE = (100, 300, 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl",
        GOC / "dev" / "tap_dev_trake.jsonl"])
    ap.add_argument("--be", type=int, nargs="*", default=list(CAC_BE))
    ap.add_argument("--chi-tiet", type=int, default=None,
                    help="in hiệu TỪNG CÂU của bể này so với bể đầu")
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
    print(f"\n{len(giu)} câu TRAKE | kênh 3 w={W3:g}\n")

    nho = {}

    def ung_vien(c, be):
        if (c.id, be) not in nho:
            ds = []
            for sk in R.tach_su_kien(c.cau_hoi):
                anh = hop_nhat([k1.tim(m, k=be) for m in R.tach_truy_van(sk)])
                ds.append(hop_nhat([anh, k3.tim(sk, k=be)],
                                   trong_so=[1.0, W3])[:be])
            nho[(c.id, be)] = ds
        return nho[(c.id, be)]

    print("CHẨN ĐOÁN — bể ứng viên đổi thì có gì đổi")
    print(f"  {'--be':>6}{'video đủ ứng viên':>20}{'dòng nộp được':>16}")
    print("  " + "-" * 42)
    for be in a.be:
        nv, nd = [], []
        for c in giu:
            ds = ung_vien(c, be)
            nv.append(len(KB.cham_video(KB.gom_theo_video(ds))))
            nd.append(len(KB.lap_dong(ds, master, 100)))
        print(f"  {be:>6}{int(st.median(nv)):>12} (trung vị)"
              f"{int(st.median(nd)):>10}/100")
    print()

    def lam(be):
        return lambda c: KB.lap_dong(ung_vien(c, be), master, 100)

    bang = {(f"{'MỐC: ' if be == a.be[0] else ''}bể {be}"):
            cham_trake_nhieu_muc(giu, lam(be), master) for be in a.be}
    print(bao_cao_tu_bang(bang))

    # ⚠️ TỪNG CÂU, không chỉ trung bình.
    #
    # Đường cong theo `--be` KHÔNG trơn: phẳng ở 150-200 rồi nhảy ở 300. Một
    # tham số trơn mà cho bậc thang thì hoặc có ngưỡng thật, hoặc trung bình
    # đang bị một hai câu kéo. Hai khả năng đó phân biệt được bằng cách nhìn
    # phân bố hiệu, và chỉ bằng cách đó.
    if a.chi_tiet and len(a.be) >= 2:
        goc, doi = a.be[0], a.chi_tiet
        b0 = bang[f"MỐC: bể {goc}"][2.0].set_index("id").diem
        b1 = bang[f"bể {doi}"][2.0].set_index("id").diem
        d = (b1 - b0).sort_values()
        print(f"\nTỪNG CÂU — bể {doi} so với bể {goc}, ở ±2s")
        print(f"  {'câu':<22}{'bể ' + str(goc):>10}{'bể ' + str(doi):>10}"
              f"{'hiệu':>10}")
        print("  " + "-" * 52)
        for i, h in d.items():
            if abs(h) > 1e-9:
                print(f"  {i:<22}{b0[i]:>10.4f}{b1[i]:>10.4f}{h:>+10.4f}")
        n_doi = int((d.abs() > 1e-9).sum())
        lon = d.abs().max()
        print(f"\n  {n_doi}/{len(d)} câu đổi | hiệu lớn nhất {lon:.4f} | "
              f"tổng hiệu {d.sum():+.4f}")
        print(f"  câu đóng góp nhiều nhất chiếm "
              f"{lon / abs(d.sum()) * 100:.0f}% tổng hiệu")


if __name__ == "__main__":
    main()
