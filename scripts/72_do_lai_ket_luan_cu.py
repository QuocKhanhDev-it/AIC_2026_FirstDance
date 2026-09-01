"""
72_do_lai_ket_luan_cu.py — Đo lại những thứ bị BÁC dưới mốc nền CŨ và YẾU HƠN.

    python scripts/72_do_lai_ket_luan_cu.py

VÌ SAO ĐÁNG ĐO LẠI

Ba kết luận dưới đây đều được đo khi kênh 1 còn là SigLIP2 (hoặc CLIP), và
trước A51 — tức truy vấn còn bị cắt cụt ở token 64. Từ đó tới nay kênh 1 mạnh
lên **2,3 lần** (A47) rồi mạnh thêm nữa (A51). Một kênh phụ bị bác vì "pha
loãng kênh 1" có thể đổi kết luận theo CẢ HAI chiều:

  * kênh 1 mạnh hơn -> kênh phụ càng dễ pha loãng -> càng bị bác
  * nhưng cũng: kênh 1 mạnh hơn -> RRF có nền tốt hơn -> kênh phụ chỉ cần
    đúng ở vài câu kênh 1 trượt là đã có lãi

Không đoán được chiều nào. Đo.

BA THỨ ĐEM ĐO LẠI

**Kênh 2 — metadata cấp video** (A12: được 0,0000 ở ±2s nên mặc định BỎ;
A14.2: cộng vào là pha loãng). Cả hai đo trên tập dev TỰ SOẠN, thứ mà A50 chứng
minh là thổi phồng kênh 1 gấp 2,3 lần.

**Kênh 4 — objects + IDF** (A25: kênh 3 mạnh gấp 2,8 lần). Đó là so hai kênh
ĐỨNG RIÊNG, không phải đo objects THÊM vào cấu hình hiện tại. Yếu hơn không có
nghĩa là thừa — RRF cần kênh **bổ sung**, không cần kênh mạnh.

**Ràng buộc đa dạng `moi_video`** (A18: làm tệ đi). Đo khi kênh 1 là SigLIP2,
lúc bể ứng viên vón cục quanh vài video. Giờ A55 đo được top-20 đã trải trên
10,2 video — ràng buộc này có thể đã thành thừa, hoặc thành có ích, chưa biết.

⚠️ Mốc nền là cấu hình `run.py` ĐANG chạy sau A52, không phải cấu hình mà các
kết luận cũ dùng làm mốc.
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
from objects import KenhObjects                       # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
W_PHU = (0.25, 0.5)


def gioi_han_video(ds, n: int):
    """Giữ tối đa `n` ứng viên mỗi video, theo thứ hạng."""
    dem, ra = {}, []
    for c in ds:
        if dem.get(c.video_id, 0) < n:
            dem[c.video_id] = dem.get(c.video_id, 0) + 1
            ra.append(c)
    return ra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    k2 = KenhVanBan.tu_metadata(master)
    k4 = KenhObjects(str(a.index), master)

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu\n")

    nen = {}

    def phan(c):
        """Bốn danh sách gốc, tính một lần cho mọi cấu hình."""
        if c.id not in nen:
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
            # Kênh 2 là kênh CẤP VIDEO: không giới hạn thì một video khớp đổ cả
            # trăm keyframe vào và chiếm sạch top-100 (bm25.tim docstring).
            nen[c.id] = (anh, k3.tim(c.cau_hoi, k=a.be),
                         k2.tim(c.cau_hoi, k=a.be, moi_video=3),
                         k4.tim(c.cau_hoi, k=a.be))
        return nen[c.id]

    def _nho(f):
        n = {}

        def g(c):
            if c.id not in n:
                n[c.id] = f(c)[:100]
            return n[c.id]
        return g

    def moc(c):
        anh, ocr, _, _ = phan(c)
        return hop_nhat([anh, ocr], trong_so=[1.0, W3])

    cau_hinh = {"1. mốc: run.py (A52)": _nho(moc)}

    for w in W_PHU:
        cau_hinh[f"2. + kênh 4 objects ({w:g})"] = _nho(
            (lambda w: lambda c: hop_nhat(
                [phan(c)[0], phan(c)[1], phan(c)[3]],
                trong_so=[1.0, W3, w]))(w))
    for w in W_PHU:
        cau_hinh[f"3. + kênh 2 metadata ({w:g})"] = _nho(
            (lambda w: lambda c: hop_nhat(
                [phan(c)[0], phan(c)[1], phan(c)[2]],
                trong_so=[1.0, W3, w]))(w))
    cau_hinh["4. + cả kênh 2 và 4 (0,25)"] = _nho(lambda c: hop_nhat(
        list(phan(c)), trong_so=[1.0, W3, 0.25, 0.25]))
    for n in (3, 5):
        cau_hinh[f"5. mỗi video tối đa {n} dòng"] = _nho(
            (lambda n: lambda c: gioi_han_video(moc(c), n))(n))

    # Chẩn đoán: hai kênh này đứng một mình được bao nhiêu, ĐO LẠI trên đề thật
    cau_hinh["6. chỉ kênh 4 (chẩn đoán)"] = _nho(lambda c: phan(c)[3])
    cau_hinh["7. chỉ kênh 2 (chẩn đoán)"] = _nho(lambda c: phan(c)[2])

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
