"""
79_do_dap_an_qa.py — Mỗi dòng Q&A một `answer` riêng, thay vì một chuỗi dùng chung.

    python scripts/79_do_dap_an_qa.py --file <tap>.jsonl

⚠️ PHÁT HIỆN QUAN TRỌNG HƠN CHÍNH ĐỀ XUẤT

`cham_diem._dung_dap_an` coi ứng viên **không có** khoá `answer` là HỢP LỆ — cố
ý, vì "lúc này ta đang đo TRUY HỒI". Mà mọi kênh đều không gắn `answer`.

Nghĩa là **mọi điểm Q&A trong repo đều được chấm như thể đáp án luôn đúng.**
BTC thì cho 0 điểm nếu `answer` sai hoặc trống. Điểm Q&A ta đang báo là TRẦN
TRÊN, không phải điểm thi.

Và `run.py` hiện chỉ có `--tra-loi`: **một chuỗi dùng chung cho cả 100 dòng**.
Nên trong bài nộp thật, hoặc chuỗi đó đúng (mọi dòng đều có cơ hội) hoặc sai
(cả 100 dòng đều 0 điểm), bất kể truy hồi tốt đến đâu.

BỐN CẤU HÌNH

  1. `answer` bỏ trống  — cách repo ĐANG chấm. Trần trên, không phải điểm thi.
  2. một chuỗi dùng chung, đào từ khung hạng 1 — cách `run.py` đang nộp
  3. mỗi dòng một `answer`, đào từ OCR/ASR CỦA CHÍNH KHUNG ĐÓ — đề xuất
  4. `answer` đúng ở mọi dòng (oracle) — bằng cấu hình 1, để đối chiếu

BỘ ĐÀO ĐÁP ÁN

Đọc loại đáp án cần tìm từ câu hỏi (số / tên riêng), rồi lấy ứng viên từ
`ocr_text` + `asr_text` của chính keyframe đó. Không có gì thông minh — đúng
tinh thần "đo trước, tinh vi sau".
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import (DUNG_SAI_CHINH, DUNG_SAI_KIEM,  # noqa: E402
                       diem_cau, no_cua_so)
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
SO = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\s?(?:g|kg|ml|l|%)?\b", re.I)
# Tên riêng tiếng Việt: 1–3 từ viết hoa liên tiếp
TEN = re.compile(r"\b[A-ZĐÂÊÔƯĂÁÀÃẢẠÉÈẼẺẸÍÌĨỈỊÓÒÕỎỌÚÙŨỦỤÝỲỸỶỴ]"
                 r"[a-zàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớ"
                 r"ờởỡợụủứừửữựỳỵỷỹ]+(?:\s+[A-ZĐÂÊÔƯĂ][a-zàáâãèéêìíòóôõùúý"
                 r"ăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+){0,2}\b")
TU_HOI_SO = re.compile(r"\b(bao nhiêu|mấy|số|khối lượng|gam|kg|phần trăm|"
                       r"nhiệt độ|năm|giờ|độ)\b", re.I)


def bo_dau(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


DUNG = set("la gi cua o trong va co nao bao nhieu may mot hai ba cho nguoi".split())


def dao_dap_an(van: str, hoi_so: bool, cau_hoi: str = "") -> str:
    """Một chuỗi ứng viên từ văn bản của chính keyframe. '' nếu không có.

    `cau_hoi` rỗng -> lấy ứng viên ĐẦU TIÊN (bản ngây thơ).
    Có `cau_hoi` -> chọn ứng viên GẦN NHẤT với một từ khoá của câu hỏi trong
    văn bản. Cần thiết vì OCR bản tin đầy dấu thời gian ("06:30:11") và số
    hiệu kênh — lấy số đầu tiên gần như luôn trúng chúng.
    """
    if not van:
        return ""
    uv = ([(m.group().strip(), m.start()) for m in SO.finditer(van)] if hoi_so
          else [(m.group().strip(), m.start()) for m in TEN.finditer(van)
                if len(m.group()) > 2])
    if not uv:
        return ""
    if not cau_hoi:
        return uv[0][0]

    v = bo_dau(van)
    khoa = [w for w in bo_dau(cau_hoi).split()
            if len(w) > 2 and w not in DUNG]
    vi_tri = [m.start() for w in khoa for m in re.finditer(re.escape(w), v)]
    if not vi_tri:
        return uv[0][0]
    return min(uv, key=lambda x: min(abs(x[1] - p) for p in vi_tri))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    van_theo_row = {}
    for r, o, s in zip(bang.row_id.values, bang.ocr_text.fillna("").values,
                       bang.asr_text.fillna("").values):
        van_theo_row[int(r)] = f"{o} {s}".strip()

    cau = [c for c in tap_dev.doc(a.file) if c.loai == "QA" and c.dap_an]
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)}/{len(cau)} câu Q&A đo được\n")

    # Bộ đào có tìm ra đáp án ở CHÍNH khung đáp án không? (kiểu A26)
    trung = 0
    for c in giu:
        hoi_so = bool(TU_HOI_SO.search(c.cau_hoi))
        van = " ".join(van_theo_row.get(r, "") for r in c.row_id_dung)
        if bo_dau(c.dap_an) in bo_dau(van):
            trung += 1
    print(f"đáp án XUẤT HIỆN trong OCR/ASR của khung đúng: {trung}/{len(giu)}"
          f"  <- trần trên của MỌI cách đào từ văn bản\n")

    for ds_giay in (DUNG_SAI_CHINH, DUNG_SAI_KIEM):
        print(f"── dung sai ±{ds_giay:g}s " + "─" * 40)
        print(f"{'câu':<16}{'đáp án':<16}{'BỎ TRỐNG':>10}{'CHUNG':>8}{'RIÊNG':>8}{'GẦN KHOÁ':>10}")
        tong = [0.0, 0.0, 0.0, 0.0]
        for c in giu:
            hoi_so = bool(TU_HOI_SO.search(c.cau_hoi))
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
            ds = hop_nhat([anh, k3.tim(c.cau_hoi, k=a.be)],
                          trong_so=[1.0, W3])[:100]
            dung = no_cua_so(c.row_id_dung, master, ds_giay)
            mong = c.dap_an.strip().lower()

            # 1. bỏ trống -> mọi dòng hợp lệ (cách repo đang chấm)
            h1 = next((i for i, x in enumerate(ds, 1) if x.row_id in dung), None)
            # 2. một chuỗi dùng chung, đào từ khung hạng 1
            chung = dao_dap_an(van_theo_row.get(ds[0].row_id, ""), hoi_so) if ds else ""
            h2 = h1 if chung.strip().lower() == mong else None
            # 3. mỗi dòng một đáp án, đào từ chính khung đó
            h3 = next((i for i, x in enumerate(ds, 1)
                       if x.row_id in dung
                       and dao_dap_an(van_theo_row.get(x.row_id, ""),
                                      hoi_so).strip().lower() == mong), None)
            h4 = next((i for i, x in enumerate(ds, 1)
                       if x.row_id in dung
                       and dao_dap_an(van_theo_row.get(x.row_id, ""), hoi_so,
                                      c.cau_hoi).strip().lower() == mong), None)
            d = [diem_cau(h1), diem_cau(h2), diem_cau(h3), diem_cau(h4)]
            tong = [x + y for x, y in zip(tong, d)]
            print(f"{c.id:<16}{c.dap_an[:14]:<16}"
                  f"{d[0]:>10.4f}{d[1]:>8.4f}{d[2]:>8.4f}{d[3]:>10.4f}")
        n = len(giu)
        print(f"{'TRUNG BÌNH':<32}{tong[0]/n:>10.4f}{tong[1]/n:>8.4f}"
              f"{tong[2]/n:>8.4f}{tong[3]/n:>10.4f}\n")


if __name__ == "__main__":
    main()
