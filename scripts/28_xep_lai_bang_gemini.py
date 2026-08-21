"""
28_xep_lai_bang_gemini.py — Xếp lại top-N ứng viên bằng Gemini đọc OCR/ASR.

MÔ HÌNH LỌC XẾP TẦNG, KHÔNG PHẢI HỢP NHẤT NGANG HÀNG
=====================================================

A14/A17/A22 đã bác bốn lần việc cho kênh yếu bỏ phiếu ngang hàng với kênh mạnh:
RRF cộng `1/(k+hạng)` mà không nhìn kênh nào tốt hơn, nên ứng viên hạng 1 của
kênh chết được cộng đúng bằng hạng 1 của kênh tốt.

Ở đây khác: **kênh 1 (SigLIP2) giữ quyền chọn bể**, Gemini chỉ được **xếp lại
trong bể đó**, và chỉ trên `--top` ứng viên đầu. Phần còn lại giữ nguyên thứ tự.
Không ứng viên nào bị bỏ — kịch bản xấu nhất là xáo thứ tự trong top-N.

⚠️ **CHỈ ĐẨY LÊN KHI CHẮC.** Lời nhắc buộc Gemini chỉ nêu những ứng viên có
bằng chứng RÕ trong văn bản; số còn lại giữ nguyên thứ tự cũ. Vì R@1 chiếm 1/5
tổng điểm, một lần đẩy nhầm đắt hơn nhiều một lần bỏ lỡ.

⚠️ **OCR CỦA MỘT KEYFRAME THƯỜNG KHÔNG TẢ CẢNH TRONG KEYFRAME ĐÓ.** Bản tin có
dòng chữ chạy về một tin khác hẳn; bài giảng có công thức Toán. Nên với truy vấn
thuần thị giác ("người phụ nữ thái cà chua"), văn bản gần như vô dụng và Gemini
phải trả về danh sách rỗng. Đó là hành vi ĐÚNG, không phải hỏng.

    python scripts/28_xep_lai_bang_gemini.py --nguon FirstDance_round1.zip \\
        --de dev/THUNGHIEM-bo-de-thi --ra submission_v6 --nen firstdance5.zip

⚠️ Chưa đo được trên tập dev: bể ứng viên ở đây do SigLIP2 sinh ra, mà máy này
không chạy nổi SigLIP2 (A25). Đo trên bể kênh 3 là đo một chế độ khác. Nên đây
là **phép thử trên vòng thử nghiệm**, không phải cấu hình đã chứng minh.
"""

import argparse
import csv
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
from nop_bai import dong_goi, ghi_goi, soat_zip        # noqa: E402
from schema import AnswerKIS, AnswerQA, AnswerTRAKE    # noqa: E402
from tra_loi_ocr import MODEL_GEMINI, _goi_gemini      # noqa: E402

NHAC = """Bạn đang giúp xếp hạng lại kết quả tìm kiếm video.

TRUY VẤN:
{truy_van}

Dưới đây là {n} khung hình ứng viên. Với mỗi khung, bạn CHỈ thấy tiêu đề video,
chữ đọc được (OCR) và lời nói (ASR) quanh khung đó — bạn KHÔNG nhìn thấy hình.

{ung_vien}

Nhiệm vụ: liệt kê số thứ tự của những khung mà VĂN BẢN chứa bằng chứng RÕ RÀNG
khớp với truy vấn (tên riêng, địa danh, con số, câu chữ được nhắc trong truy vấn).

Quy tắc bắt buộc:
- Chỉ nêu khung có bằng chứng RÕ. Không đoán theo cảm tính.
- Truy vấn tả cảnh thuần thị giác mà văn bản không giúp gì -> trả về [].
- Xếp khung chắc chắn nhất lên đầu.
- Trả lời ĐÚNG một dòng JSON, không giải thích. Ví dụ: [7, 2, 15]"""


def doc_zip(z: Path) -> dict:
    ra = {}
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if not i.is_dir() and i.filename.lower().endswith(".csv"):
                ra[Path(i.filename).stem] = [
                    r for r in csv.reader(
                        io.StringIO(f.read(i.filename).decode("utf-8"))) if r]
    return ra


def van_ban_cua(bang, master, video_id: str, frame_idx: int,
                toi_da: int = 240, metadata: bool = False) -> str:
    """Bằng chứng văn bản về khung `(video_id, frame_idx)`, cắt cho vừa lời nhắc.

    `metadata=True` thêm TIÊU ĐỀ video vào trước. Vì sao đáng thêm dù A12 đo
    được kênh 2 chỉ 0,0000 ở mức khung: tiêu đề mô tả **cả video**, nên nó vô
    dụng khi phải chọn giữa các khung TRONG một video — nhưng ở đây Gemini đang
    chọn giữa các khung của **nhiều video khác nhau**, và câu hỏi "video này nói
    về cái gì" là đúng thứ tiêu đề trả lời được. Cùng một dữ liệu, đổi vai trò
    thì đổi giá trị.
    """
    g = master[(master.video_id == video_id)
               & (master.frame_idx.astype(int) == int(frame_idx))]
    if g.empty:
        return ""
    rid = int(g.row_id.iloc[0])
    x = bang.iloc[rid]
    o = " ".join(str(x.get("ocr_text", "") or "").split())
    a = " ".join(str(x.get("asr_text", "") or "").split())
    phan = []
    if metadata:
        tieu_de = " ".join(str(g.title.iloc[0] or "").split())
        if tieu_de:
            phan.append(f"VIDEO: {tieu_de[:toi_da]}")
    if o:
        phan.append(f"OCR: {o[:toi_da]}")
    if a:
        phan.append(f"ASR: {a[:toi_da]}")
    return " | ".join(phan)


def xep_lai(dong: list, truy_van: str, bang, master, top: int,
            model: str, metadata: bool = False) -> tuple[list, list]:
    """Trả `(thứ tự mới, các vị trí được đẩy lên)`. Không bỏ dòng nào."""
    dau = dong[:top]
    mo_ta = []
    for i, r in enumerate(dau, 1):
        vb = van_ban_cua(bang, master, r[0], r[1], metadata=metadata)
        mo_ta.append(f"[{i}] {r[0]} frame {r[1]} — {vb or '(không có chữ)'}")

    tho = _goi_gemini(NHAC.format(truy_van=truy_van, n=len(dau),
                                  ung_vien="\n".join(mo_ta)), model=model)
    m = re.search(r"\[[\d,\s]*\]", tho or "")
    if not m:
        return dong, []
    try:
        chon = [int(x) for x in json.loads(m.group(0))]
    except (ValueError, json.JSONDecodeError):
        return dong, []
    chon = [i for i in dict.fromkeys(chon) if 1 <= i <= len(dau)]
    if not chon:
        return dong, []

    len_ = [dau[i - 1] for i in chon]
    con_lai = [r for i, r in enumerate(dau, 1) if i not in set(chon)]
    return len_ + con_lai + dong[top:], chon


def main():
    ap = argparse.ArgumentParser(description="xep lai top-N bang Gemini + OCR")
    ap.add_argument("--nguon", required=True, type=Path)
    ap.add_argument("--de", required=True, type=Path)
    ap.add_argument("--ra", default=Path("submission_xeplai"), type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--top", type=int, default=20,
                    help="chỉ xếp lại ngần này ứng viên đầu")
    ap.add_argument("--model", default=MODEL_GEMINI)
    ap.add_argument("--metadata", action="store_true",
                    help="thêm TIÊU ĐỀ video vào bằng chứng cho mỗi ứng viên")
    ap.add_argument("--nen", metavar="FILE.zip")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    de = R.doc_de(a.de)
    tho = doc_zip(a.nguon)
    print(f"{a.nguon}: {len(tho)} gói | xếp lại top-{a.top} bằng {a.model}\n")

    goi, so_su_kien = {}, {}
    for ten, dong in sorted(tho.items()):
        loai = ten.rsplit("-", 1)[-1]
        if loai == "trake":
            so_su_kien[ten] = len(dong[0]) - 1
            goi[ten] = [AnswerTRAKE(r[0], [int(x) for x in r[1:]]) for r in dong]
            continue
        if loai == "qa":     # đáp án Q&A giữ nguyên — xem A26
            goi[ten] = [AnswerQA(r[0], int(r[1]), r[2] if len(r) > 2 else "")
                        for r in dong]
            continue

        moi, chon = xep_lai(dong, de.get(ten, ""), bang, master, a.top,
                            a.model, a.metadata)
        goi[ten] = [AnswerKIS(r[0], int(r[1])) for r in moi]
        print(f"  {ten:<20} đẩy lên {len(chon):>2} ứng viên"
              f"{'  -> ' + str(chon[:5]) if chon else '  (văn bản không giúp gì)'}")

    if Path(a.ra).exists():
        shutil.rmtree(a.ra)
    d = ghi_goi(goi, a.ra, so_su_kien)
    print(f"\n✅ Đã ghi {d}")
    if a.nen:
        z = dong_goi(d, a.nen)
        loi, canh = soat_zip(z)
        for x in canh:
            print("⚠️ ", x)
        if loi:
            print(f"❌ {len(loi)} lỗi:")
            for x in loi:
                print("   ", x)
            raise SystemExit(1)
        print(f"✅ Đã nén -> {z} — đạt checklist BTC")


if __name__ == "__main__":
    main()
