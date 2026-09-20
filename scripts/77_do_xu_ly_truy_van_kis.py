"""
77_do_xu_ly_truy_van_kis.py — Ba đề xuất về cách xử lý truy vấn KIS.

    python scripts/77_do_xu_ly_truy_van_kis.py --file <tap>.jsonl

BA THAY ĐỔI, ĐO RIÊNG TỪNG CÁI

**A. Ngưỡng tách mệnh đề.** `tach_truy_van` giữ nguyên câu nếu ≤ `TRAN_TOKEN`
từ, mặc định **40**. Đo được: 10/52 câu đề thật giữ nguyên, 42 câu bị tách,
trung vị 62 từ/câu. Đề xuất: nâng ngưỡng để giữ nguyên nhiều câu hơn, vì tách
làm mất **liên kết chủ–vị** ("người áo đỏ đi xe máy đen" tách ra thì một người
áo đỏ ở góc này + một xe máy đen ở góc kia cũng khớp).

Ngược lại là trần 64 token của tháp văn bản: giữ nguyên câu dài là bị cắt cụt,
mà A51 đo được cắt cụt còn HẠI hơn (thêm cả câu vào RRF: 0,4875 so với 0,5096).

**B. Trọng số mệnh đề theo ĐỘ HIẾM TỪ.** RRF hiện cào bằng mọi mệnh đề. Mệnh đề
"ngoài trời, ban ngày" và mệnh đề "cầm con vẹt màu đỏ" có tiếng nói ngang nhau.
Cân theo độ hiếm: mệnh đề toàn từ phổ biến -> trọng số thấp.

A55 đã bác "đồng thuận mệnh đề" (đếm SỐ mệnh đề trúng, −0,0163). Đây là thứ
khác: không đếm số lượng, mà cân CHẤT LƯỢNG từng mệnh đề.

**C. Cổng kênh 3 theo từ khoá chỉ thị.** Hiện kênh 3 luôn chạy ở trọng số 0,5.
Câu tả thị giác thuần ("flycam quay cánh đồng lúa") thì ASR chỉ là nhiễu.

⚠️ A58 đã bác một thứ GẦN GIỐNG: hạn chế ĐẦU VÀO của kênh 3 xuống mấy mệnh đề
có tín hiệu chữ (−0,0202). Cái này khác: đầu vào vẫn là CẢ CÂU, chỉ bật/tắt
TRỌNG SỐ. Đáng đo vì A58 kết luận "kênh 3 cần cả câu" — nó không nói gì về việc
có nên tắt hẳn kênh 3 ở câu thị giác thuần hay không.
"""

import argparse
import math
import re
import sys
import unicodedata
from collections import Counter
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
NGUONG_TACH = (40, 50, 60, 10_000)          # 10.000 = không bao giờ tách
# Từ chỉ thị có CHỮ trên màn hình hoặc LỜI nói -> kênh 3 mới có việc
CHI_THI = re.compile(
    r"\b(chữ|dòng chữ|biển|bảng|ghi|đề|tiêu đề|nhãn|băng.?rôn|khẩu hiệu|"
    r"biển số|nói|phát biểu|cho biết|kể|giới thiệu|trả lời|phỏng vấn|hát|"
    r"tiếng|lời)\b|[\"“”']", re.I)


def bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def bang_tan_suat(bang: pd.DataFrame) -> dict:
    """Tần suất từ trong kho OCR/ASR — dùng làm thước 'phổ biến'."""
    d = Counter()
    for x in bang["text"].fillna(""):
        d.update(bo_dau(x).split())
    return d


def trong_so_menh_de(md: list, tan: Counter, tong: int, cach: str = "max",
                     idf_giua: float = 1.0) -> list:
    """Trọng số mỗi mệnh đề theo độ hiếm từ. Chuẩn hoá về [0,2 ; 1,0].

    `cach="max"`: lấy IDF của từ HIẾM NHẤT trong mệnh đề.
        Chỉ cần mệnh đề chứa một thực thể đặc thù là nó xứng đáng có tiếng nói.
    `cach="tb"`: IDF trung bình — phạt oan mệnh đề ngắn có 1 từ hiếm + 1 từ
        rác, và cũng để mệnh đề dài rườm rà tự pha loãng chính nó.

    ⚠️ CHẶN TRẦN. Từ gõ sai hoặc từ chưa từng có trong kho ASR cho IDF cực đại,
    khiến một mệnh đề chiếm trọn trọng số và phá cả RRF. Chia cho IDF trung vị
    rồi kẹp về [0,2 ; 1,0] — mệnh đề chung chung nhận 0,2, mệnh đề hiếm nhận
    1,0, không ai nhận 12,0 vì một lỗi chính tả.
    """
    ra = []
    for m in md:
        tu = [t for t in bo_dau(m).split() if len(t) > 1]
        if not tu:
            ra.append(1.0)
            continue
        idf = [math.log(tong / (1 + tan.get(t, 0))) for t in tu]
        x = max(idf) if cach == "max" else sum(idf) / len(idf)
        ra.append(min(1.0, max(0.2, x / idf_giua)))
    return ra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")

    print("dựng bảng tần suất từ kho OCR/ASR…", flush=True)
    tan = bang_tan_suat(bang)
    tong = sum(tan.values())
    # IDF trung vị của chính kho — mốc để chuẩn hoá, tránh "runaway weight".
    _idf = sorted(math.log(tong / (1 + v)) for v in tan.values())
    idf_giua = _idf[len(_idf) // 2]
    print(f"  {len(tan):,} từ khác nhau, {tong:,} lượt\n")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    n_chi_thi = sum(1 for c in giu if CHI_THI.search(c.cau_hoi))
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu | "
          f"{n_chi_thi} câu có từ chỉ thị chữ/lời "
          f"({n_chi_thi / len(giu) * 100:.0f}%)\n")

    ocr, anh_goc = {}, {}

    def kenh3(c):
        if c.id not in ocr:
            ocr[c.id] = k3.tim(c.cau_hoi, k=a.be)
        return ocr[c.id]

    def anh(c, tran=None, can=None):
        khoa = (c.id, tran, can)
        if khoa not in anh_goc:
            md = R.tach_truy_van(c.cau_hoi, tran_tu=tran or R.TRAN_TOKEN)
            thieu = k1.co_du(md)
            if thieu:                       # ngưỡng khác -> mệnh đề khác
                anh_goc[khoa] = None
            else:
                ds = [k1.tim(m, k=a.be) for m in md]
                ts = (trong_so_menh_de(md, tan, tong, can, idf_giua)
                      if can else [1.0] * len(ds))
                anh_goc[khoa] = hop_nhat(ds, trong_so=ts)
        return anh_goc[khoa]

    def _nho(f):
        n = {}

        def g(c):
            if c.id not in n:
                n[c.id] = f(c)[:100]
            return n[c.id]
        return g

    def moc(c, tran=None, can=None, cong=None):
        av = anh(c, tran, can)
        if av is None:
            av = anh(c)                     # thiếu cache -> giữ mốc, không bịa
        w = W3
        if cong is not None and not CHI_THI.search(c.cau_hoi):
            w = cong                      # sàn cho câu thị giác thuần
        return hop_nhat([av, kenh3(c)], trong_so=[1.0, w]) if w else av

    cau_hinh = {"0. mốc: run.py (ngưỡng 40)": _nho(moc)}
    for t in NGUONG_TACH[1:]:
        ten = "không tách" if t > 1000 else f"ngưỡng {t} từ"
        cau_hinh[f"A. {ten}"] = _nho((lambda t: lambda c: moc(c, tran=t))(t))
    for cach, ten in (("max", "MAX IDF"), ("tb", "IDF trung bình")):
        cau_hinh[f"B. cân mệnh đề — {ten}"] = _nho(
            (lambda k: lambda c: moc(c, can=k))(cach))
    for san in (0.0, 0.1, 0.25):
        cau_hinh[f"C. cổng kênh 3 — sàn {san:g}"] = _nho(
            (lambda s: lambda c: moc(c, cong=s))(san))
    cau_hinh["B+C (MAX IDF + sàn 0,1)"] = _nho(
        lambda c: moc(c, can="max", cong=0.1))

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
