"""
104_doi_chieu_diem_that.py — Điểm THẬT của BTC so với thứ hạng hệ thống cho ra.

    python scripts/104_doi_chieu_diem_that.py

Đây là lần đầu repo có **nhãn vàng thật**: bảng điểm từng câu của bài nộp Sơ
tuyển 1 (14,5/25). Mọi con số trước nay đo trên đáp án TỰ SOI — nhãn của BTC là
thứ duy nhất kiểm được cách soi đó có đúng không.

HAI CÂU HỎI, VÀ CHÚNG DẪN TỚI HAI VIỆC KHÁC HẲN NHAU

Với mỗi câu BTC cho 0 điểm:

  * đáp án **NGOÀI bể** -> giới hạn của MODEL. Xếp lại bao nhiêu cũng vô ích,
    phải đổi/bổ sung tín hiệu.
  * đáp án **TRONG bể nhưng xếp thấp** -> giới hạn của XẾP LẠI. Đây là khoảng
    trống 0,3337 của A54.

Trộn hai nhóm rồi kết luận chung là cách chắc chắn nhất để đầu tư nhầm chỗ.

⚠️ BA CẢNH BÁO KHI ĐỌC BẢNG NÀY

**1. Đáp án của ta là TỰ SOI, không phải của BTC.** Nếu ta soi trúng một
khoảnh khắc KHÁC với khoảnh khắc BTC chấm, thì "hạng" dưới đây đo nhầm mục
tiêu. Câu nào điểm thật và hạng đo LỆCH NHAU MẠNH là ứng viên số một cho lỗi
loại này — và đó là thông tin, không phải nhiễu.

**2. Bài nộp thật chạy CẤU HÌNH CŨ.** Từ đó tới nay đã đổi: RRF hạng cho mệnh
đề (A51), trọng số kênh 3 = 0,5 (A52), K-best cho TRAKE (A79). Bảng này đo hệ
thống HIỆN TẠI, nên chênh lệch với điểm thật gồm cả phần đã cải thiện.

**3. Chỉ có nhãn cho 20/25 câu.** Năm câu còn lại (p1-3, p1-19, p1-21, p1-22)
chưa soi ra đáp án.
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
from cham_diem import DUNG_SAI_CHINH, no_cua_so       # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5

# Bảng điểm BTC — Sơ tuyển 1, tổng 14,5/25. Khoá theo SỐ GÓI, vì hậu tố loại
# trong bảng chép tay lệch vài chỗ so với tên file thật trong `de_thi_thu/`.
DIEM_THAT = {
    "p1-1": 1.0, "p1-2": 0.5, "p1-3": 0.0, "p1-4": 1.0, "p1-5": 1.0,
    "p1-6": 1.0, "p1-7": 0.5, "p1-8": 1.0, "p1-9": 1.0, "p1-10": 1.0,
    "p1-11": 0.0, "p1-12": 0.0, "p1-13": 0.0, "p1-14": 1.0, "p1-15": 1.0,
    "p1-16": 0.0, "p1-17": 1.0, "p1-18": 0.0, "p1-19": 0.0, "p1-20": 1.0,
    "p1-21": 0.5, "p1-22": 0.0, "p1-23": 0.0, "p1-24": 1.0, "p1-25": 1.0,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_thi_thu.jsonl",
                    type=Path)
    ap.add_argument("--be", type=int, default=1000)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = tap_dev.doc(a.file)
    print(f"{len(cau)}/25 câu có nhãn tự soi | bể {a.be}\n")
    print(f"{'gói':<10}{'loại':<7}{'BTC':>6}{'hạng của đáp án':>18}"
          f"{'trong top':>11}   nhận định")
    print("-" * 78)

    nhom = {"model": [], "xep_lai": [], "khop": []}
    for c in sorted(cau, key=lambda x: int(x.id.rsplit("-", 1)[1])):
        goi = "p1-" + c.id.rsplit("-", 1)[1]
        that = DIEM_THAT.get(goi)

        if c.loai == "TRAKE":
            moc = [m for b in c.row_id_dung for m in (b if isinstance(b, list) else [b])]
            sk = R.tach_su_kien(c.cau_hoi)
            uv = hop_nhat([hop_nhat([k1.tim(m, k=a.be)
                                     for m in R.tach_truy_van(sk[0])]),
                           k3.tim(sk[0], k=a.be)], trong_so=[1.0, W3])
        else:
            moc = c.row_id_dung
            uv = hop_nhat([hop_nhat([k1.tim(m, k=a.be)
                                     for m in R.tach_truy_van(c.cau_hoi)]),
                           k3.tim(c.cau_hoi, k=a.be)], trong_so=[1.0, W3])

        dung = no_cua_so(moc, master, DUNG_SAI_CHINH)
        h = next((i for i, x in enumerate(uv[:a.be], 1) if x.row_id in dung), None)

        if h is None:
            nx, khoa = "NGOÀI bể — giới hạn MODEL", "model"
        elif h <= 20:
            nx, khoa = "trong top-20", "khop"
        else:
            nx, khoa = "trong bể, xếp thấp — giới hạn XẾP LẠI", "xep_lai"
        nhom[khoa].append((goi, that, h))

        print(f"{goi:<10}{c.loai:<7}{that if that is not None else '?':>6}"
              f"{(h if h else 'ngoài bể'):>18}"
              f"{('top-' + str(min(x for x in (1,5,20,50,100) if h and h <= x)) if h and h <= 100 else '—'):>11}"
              f"   {nx}")

    print()
    for khoa, ten in (("model", "NGOÀI bể (giới hạn MODEL)"),
                      ("xep_lai", "trong bể, hạng > 20 (giới hạn XẾP LẠI)"),
                      ("khop", "trong top-20")):
        ds = nhom[khoa]
        n0 = sum(1 for _, t, _ in ds if t == 0.0)
        print(f"  {ten:<40}{len(ds):>3} câu   (BTC cho 0 điểm: {n0})")


if __name__ == "__main__":
    main()
