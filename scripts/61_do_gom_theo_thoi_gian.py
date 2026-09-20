"""
61_do_gom_theo_thoi_gian.py — Gom khung cùng cảnh trước khi nộp có lãi không?

    python scripts/61_do_gom_theo_thoi_gian.py

Ý TƯỞNG (học từ `grouping.py` của một nhóm khác)

Top-20 thường có 6–7 khung của **cùng một cảnh**, cách nhau vài giây. Chúng
chiếm chỗ của những cảnh KHÁC, mà bài nộp chỉ có 100 dòng và BTC tính hạng của
dòng ĐÚNG ĐẦU TIÊN. Bỏ bớt bản sao thì mọi thứ phía sau **dịch lên**, và đáp án
nếu nằm dưới sẽ lên hạng.

Gom: duyệt theo thứ hạng, giữ một ứng viên trừ khi đã giữ một ứng viên khác
**cùng video, cách dưới N giây**. Người đại diện luôn là kẻ hạng cao nhất, nên
không có chuyện thay bằng khung xấu hơn.

⚠️ KHÁC với dedup đã bị bác ở A11. A11 gom theo **độ giống ảnh** và đo trên một
kênh đang hỏng. Ở đây gom theo **thời gian**, tiêu chí khách quan, không cần
model, và đo trên cấu hình mạnh nhất hiện có.

⚠️ HAI CHIỀU CÓ THỂ THUA. Cửa sổ chấm của BTC rộng tới vài phút (A9). Nếu
người đại diện lệch ra ngoài cửa sổ mà kẻ bị bỏ thì nằm trong, ta MẤT câu đó.
Vì vậy phải đo ở cả hai mức dung sai — cửa càng rộng, gom càng dễ có lãi; cửa
hẹp thì gom mạnh tay là tự bắn vào chân.
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
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5                                              # trọng số kênh 3 (A52)
CUA_SO = (2.0, 5.0, 15.0, 60.0)                       # giây


def gom(ds, giay: float):
    """Giữ thứ hạng, bỏ ứng viên trùng cảnh với một kẻ hạng cao hơn."""
    giu, moc = [], {}
    for c in ds:
        t = c.meta["pts_time"]
        cac_t = moc.setdefault(c.video_id, [])
        if any(abs(t - x) < giay for x in cac_t):
            continue
        cac_t.append(t)
        giu.append(c)
    return giu


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=300,
                    help="lấy bao nhiêu ứng viên TRƯỚC khi gom (gom xong cắt 100)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu | bể trước khi gom {a.be}\n")

    tho = {}

    def bo_tho(c):
        """Bể ứng viên gốc — tính một lần, mọi cấu hình dùng chung."""
        if c.id not in tho:
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
            tho[c.id] = hop_nhat([anh, k3.tim(c.cau_hoi, k=a.be)],
                                 trong_so=[1.0, W3])
        return tho[c.id]

    # Mốc: đúng `run.py` — không gom, cắt 100.
    cau_hinh = {"1. mốc: không gom (run.py)": lambda c: bo_tho(c)[:100]}
    for g in CUA_SO:
        cau_hinh[f"2. gom trong {g:g}s"] = (
            lambda g: lambda c: gom(bo_tho(c), g)[:100])(g)
    cau_hinh["3. mỗi video 1 dòng"] = lambda c: gom(bo_tho(c), 10 ** 9)[:100]

    # In trước xem mỗi cách bỏ đi bao nhiêu — cách "thắng" mà chẳng bỏ gì thì
    # chỉ là nhiễu đo.
    for ten, f in cau_hinh.items():
        n = sum(len(f(c)) for c in giu) / len(giu)
        print(f"   {ten:<28} còn {n:5.1f} dòng/câu")

    print()
    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
