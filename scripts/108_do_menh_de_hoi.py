"""
108_do_menh_de_hoi.py — Mệnh đề HỎI của câu Q&A đang làm nhiễu kênh 1?

    python scripts/108_do_menh_de_hoi.py --file dev/tap_de_that.jsonl

PHÁT HIỆN DẪN TỚI PHÉP ĐO NÀY (`107_phan_ra_diem.py`)

Phân rã 0,4769 điểm đang mất trên 52 câu nhãn sạch:

    KIS    37 câu  điểm 0,5838  mất 0,2962   R@1 0,24
    QA     12 câu  điểm 0,3500  mất 0,1500   R@1 0,08
    TRAKE   3 câu  điểm 0,4667  mất 0,0308

Q&A tệ hơn KIS ở MỌI mốc, và đây mới là con số chỉ tính TRUY HỒI — chưa xét
`answer` đúng hay sai. Tức Q&A hỏng ở khâu **tìm khung**, không chỉ ở khâu đọc
đáp án như A88 đã tập trung vào.

CƠ CHẾ

Mọi câu Q&A đều kết thúc bằng một mệnh đề HỎI, và `tach_truy_van` cắt nó ra
thành một mệnh đề riêng có **trọng số ngang** mệnh đề tả cảnh:

    | Đoạn video mô tả quá trình làm bánh, bánh có màu tím...   <- tả cảnh
    | Mỗi lần khuôn này làm được bao nhiêu cái bánh?            <- HỎI

Mệnh đề thứ hai nói về **thứ cần trả lời**, không nói **cảnh trông thế nào**.
Đem nó đi tìm ảnh thì nó kéo về bất cứ gì, mà RRF hạng lại cho nó tiếng nói
ngang mệnh đề tốt.

⚠️ CHỈ SỬA KÊNH 1. Với kênh 3 thì mệnh đề hỏi có ích thật: "biển báo", "cây
cầu", "đường nào" là từ nội dung khớp được với OCR/ASR. Bỏ nó khỏi kênh văn bản
là vứt tín hiệu.

⚠️ VÀ KHÔNG ĐỘNG TỚI ĐƯỜNG ĐÀO ĐÁP ÁN — `dap_an.py` cần đúng mệnh đề hỏi đó.

⚠️ DỰ ĐOÁN GHI TRƯỚC: bỏ hẳn sẽ hơn hạ trọng số, vì RRF hạng không nhạy với
trọng số nhỏ (A86). Nhưng n = 12 câu Q&A nên ngưỡng nhiễu sẽ RẤT rộng — nhiều
khả năng ra 🟡 dù đúng hướng. Bảng in kèm cột QA riêng để đọc được điều đó.
"""

import argparse
import re
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

W3 = 0.5

# Dấu hiệu mệnh đề HỎI tiếng Việt. Dấu `?` là dấu hiệu mạnh nhất nhưng không
# đủ: đề thật có câu hỏi viết dưới dạng mệnh lệnh ("Hãy cho biết...").
TU_HOI = re.compile(
    r"\b(bao nhiêu|mấy|là gì|màu gì|nào|thế nào|ra sao|tại sao|vì sao|"
    r"khi nào|ở đâu|ai là|hãy cho biết|hỏi )\b", re.IGNORECASE)


def la_menh_de_hoi(m: str) -> bool:
    """Mệnh đề này hỏi về ĐÁP ÁN chứ không tả CẢNH?"""
    return m.rstrip().endswith("?") or bool(TU_HOI.search(m))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--loai", default=None,
                    help="chỉ đo một loại câu (KIS/QA/TRAKE). Dùng để NHÂN BẢN "
                         "trên tập dev, nơi có 74 câu Q&A — KHÔNG dùng để lấy "
                         "con số chính: câu tập dev tự soạn ngắn hơn đề thật "
                         "và đã năm lần làm lệch kết luận (A19/A20/A31/A34/A37)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    if a.loai:
        giu = [c for c in giu if c.loai == a.loai.upper()]

    # Bao nhiêu câu THẬT SỰ bị ảnh hưởng? Nếu con số này nhỏ thì mọi hiệu ứng
    # đo được sau đây đều bị pha loãng, và phải nói rõ điều đó.
    dinh = {"KIS": 0, "QA": 0, "TRAKE": 0}
    tong_loai = {"KIS": 0, "QA": 0, "TRAKE": 0}
    for c in giu:
        me = R.tach_truy_van(c.cau_hoi)
        tong_loai[c.loai] = tong_loai.get(c.loai, 0) + 1
        if len(me) > 1 and any(la_menh_de_hoi(m) for m in me):
            dinh[c.loai] = dinh.get(c.loai, 0) + 1
    print(f"\n{len(giu)} câu | kênh 3 w={W3:g}")
    print("câu có ≥1 mệnh đề HỎI tách riêng (chỉ những câu này đổi kết quả):")
    for t in ("KIS", "QA", "TRAKE"):
        if tong_loai.get(t):
            print(f"  {t:<6} {dinh.get(t, 0):>3}/{tong_loai[t]:<3}")
    print()

    nho3, nho1 = {}, {}

    def van(c):
        if c.id not in nho3:
            nho3[c.id] = k3.tim(c.cau_hoi, k=a.be)
        return nho3[c.id]

    def anh_menh_de(m):
        """Nhớ theo MỆNH ĐỀ, không theo câu.

        Bốn cấu hình chỉ khác nhau ở cách HỢP NHẤT các mệnh đề, còn danh sách
        ứng viên của từng mệnh đề thì y hệt. Không nhớ ở đây thì mỗi mệnh đề
        bị quét brute-force 177k×1536 tới bốn lần — lần chạy đầu hết 10 phút
        mà chưa in nổi một dòng.
        """
        if m not in nho1:
            nho1[m] = k1.tim(m, k=a.be)
        return nho1[m]

    def dung(w_hoi):
        """w_hoi=None -> bỏ hẳn mệnh đề hỏi khỏi kênh 1."""
        n = {}

        def g(c):
            if c.id not in n:
                me = R.tach_truy_van(c.cau_hoi)
                if len(me) > 1:
                    ta = [m for m in me if not la_menh_de_hoi(m)]
                    hoi = [m for m in me if la_menh_de_hoi(m)]
                    if not ta:                       # toàn mệnh đề hỏi -> giữ nguyên
                        ta, hoi = me, []
                else:
                    ta, hoi = me, []
                ds = [anh_menh_de(m) for m in ta]
                ts = [1.0] * len(ds)
                if w_hoi is not None:
                    ds += [anh_menh_de(m) for m in hoi]
                    ts += [w_hoi] * len(hoi)
                anh = hop_nhat(ds, trong_so=ts)
                n[c.id] = hop_nhat([anh, van(c)], trong_so=[1.0, W3])[:100]
            return n[c.id]
        return g

    cau_hinh = {
        "1. MỐC: mệnh đề hỏi w=1 (hiện tại)": dung(1.0),
        "2. hạ trọng số w=0,5":               dung(0.5),
        "3. hạ trọng số w=0,25":              dung(0.25),
        "4. BỎ HẲN khỏi kênh 1":              dung(None),
    }
    print(bao_cao_do_nhay(giu, cau_hinh, master))

    # PHÉP THỬ ĐÚNG: chỉ các câu mà bộ nhận diện có bắn.
    #
    # Câu không có mệnh đề hỏi cho kết quả GIỐNG HỆT ở cả bốn cấu hình — chúng
    # hoà tuyệt đối, không mang thông tin nào về hiệu, nhưng vẫn bị đếm vào mẫu
    # số khi tính ngưỡng nhiễu. Đo trên tập đầy đủ là đo hiệu ứng ĐÃ PHA LOÃNG;
    # đo ở đây là đo hiệu ứng THẬT rồi tự nhân lại tỷ lệ.
    #
    # Không phải bới số: tập "câu bị ảnh hưởng" do bộ nhận diện quyết định HOÀN
    # TOÀN từ văn bản truy vấn, trước khi biết bất kỳ kết quả nào. Và 0/37 câu
    # KIS bị bắn — nhóm đối chứng có sẵn, không phải dựng thêm.
    dinh_cau = [c for c in giu
                if len(R.tach_truy_van(c.cau_hoi)) > 1
                and any(la_menh_de_hoi(m) for m in R.tach_truy_van(c.cau_hoi))]
    if dinh_cau and len(dinh_cau) < len(giu):
        ty = len(dinh_cau) / len(giu)
        gach = "=" * 70
        print(f"\n{gach}\nPHÉP THỬ CHÍNH — {len(dinh_cau)}/{len(giu)} CÂU BỘ "
              f"NHẬN DIỆN CÓ BẮN\nHiệu ở đây nhân {ty:.2f} ra hiệu trên toàn "
              f"tập.\n{gach}")
        print(bao_cao_do_nhay(dinh_cau, cau_hinh, master))


if __name__ == "__main__":
    main()
