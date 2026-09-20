"""
29_loc_cung_gemini.py — Lọc cứng OCR trên TOÀN KHO, token do Gemini chọn.

KHÁC `28_xep_lai_bang_gemini.py` Ở CHỖ CĂN BẢN
===============================================

    28  xếp lại trong bể SigLIP2 đã chọn   -> không bao giờ tìm ra khung MỚI
    29  quét toàn bộ 165.259 khung có OCR  -> đưa vào khung SigLIP2 CHƯA TỪNG thấy

Đây là A8.5 — cú pháp `/filter all ocr{hidro}` của đội AIC'25, thắng **3/5 ví dụ
thực chiến** của họ. Ta đã cài `run.py --loc-cung` từ lâu nhưng **chưa đo lần
nào**, và bản đó dùng luật đoán token hiếm quá tham: nó coi mọi chữ viết hoa là
tên riêng, mà tiếng Việt câu nào cũng viết hoa chữ đầu — nổ ở 24/24 gói đề mẫu.

Ở đây Gemini chọn token, nên nó phân biệt được *"Nguyễn Trung Trực"* (tên riêng,
sẽ hiện trên bảng hiệu) với *"Trong"* (chữ đầu câu). Rồi lọc lại bằng chính kho
OCR: token xuất hiện ở quá nhiều tài liệu thì không phải token hiếm.

    python scripts/29_loc_cung_gemini.py --nguon firstdance2.zip \\
        --de dev/THUNGHIEM-bo-de-thi --ra submission_v9 --nen firstdance8.zip

⚠️ **RỦI RO CAO HƠN 28 RẤT NHIỀU.** 28 chỉ xáo thứ tự nên xấu nhất là hoà; 29
**đẩy khung mới lên hạng 1**, tức đánh đổi hạng 1 của SigLIP2 lấy một phỏng đoán
dựa trên chữ. Nếu token không thật sự hiếm, ta vứt hạng 1 để lấy về một khung
tình cờ có chữ đó. Vì vậy ba chốt:

  * token phải xuất hiện ở `<= NGUONG_HIEM` tài liệu OCR mới được dùng
  * chèn tối đa `--toi-da` khung, mặc định 5 — đủ để đẩy lên, không đủ để chiếm
  * đòi ĐỦ mọi token (`all`), không phải bất kỳ token nào
"""

import argparse
import collections
import csv
import io
import json
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
from nop_bai import dong_goi, ghi_goi, soat_zip        # noqa: E402
from schema import AnswerKIS, AnswerQA, AnswerTRAKE    # noqa: E402
from tra_loi_ocr import MODEL_GEMINI, _goi_gemini      # noqa: E402

# Token xuất hiện ở nhiều hơn ngần này tài liệu OCR thì KHÔNG hiếm. Cùng ngưỡng
# với `run.loc_cung` để hai đường so được với nhau.
NGUONG_HIEM = 200

NHAC = """Đây là một truy vấn tìm kiếm video tiếng Việt:

{truy_van}

Hãy liệt kê những CỤM CHỮ có nhiều khả năng XUẤT HIỆN THÀNH CHỮ trên màn hình
trong đoạn video đó — biển hiệu, băng rôn, tên địa danh, tên riêng, biển số, con
số, tiêu đề bản tin.

Quy tắc bắt buộc:
- CHỈ nêu cụm thật sự đặc trưng. Bỏ qua từ thông thường ("người", "chiếc xe").
- Bỏ qua chữ chỉ vì viết hoa đầu câu.
- Không có cụm nào đặc trưng -> trả về [].
- Tối đa 3 cụm, cụm chắc chắn nhất trước.
- Trả lời ĐÚNG một dòng JSON. Ví dụ: ["Nguyễn Trung Trực", "Kiên Giang"]"""


def bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d")


def doc_zip(z: Path) -> dict:
    ra = {}
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if not i.is_dir() and i.filename.lower().endswith(".csv"):
                ra[Path(i.filename).stem] = [
                    r for r in csv.reader(
                        io.StringIO(f.read(i.filename).decode("utf-8"))) if r]
    return ra


def main():
    ap = argparse.ArgumentParser(description="loc cung OCR toan kho, token do Gemini chon")
    ap.add_argument("--nguon", required=True, type=Path)
    ap.add_argument("--de", required=True, type=Path)
    ap.add_argument("--ra", default=Path("submission_loccung"), type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--toi-da", type=int, default=5,
                    help="chèn tối đa ngần này khung mới lên đầu mỗi gói")
    ap.add_argument("--model", default=MODEL_GEMINI)
    ap.add_argument("--nen", metavar="FILE.zip")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    co = bang[bang.ocr_text.fillna("").str.strip() != ""].copy()
    co["kd"] = co.ocr_text.map(bo_dau)
    df = collections.Counter()
    for x in co.kd:
        df.update(set(re.findall(r"[^\W_]+", x)))
    print(f"{len(co):,} tài liệu OCR | ngưỡng hiếm <= {NGUONG_HIEM} tài liệu\n")

    de = R.doc_de(a.de)
    tho = doc_zip(a.nguon)
    goi, so_su_kien = {}, {}

    for ten, dong in sorted(tho.items()):
        loai = ten.rsplit("-", 1)[-1]
        if loai == "trake":
            so_su_kien[ten] = len(dong[0]) - 1
            goi[ten] = [AnswerTRAKE(r[0], [int(x) for x in r[1:]]) for r in dong]
            continue

        tho_gem = _goi_gemini(NHAC.format(truy_van=de.get(ten, "")), model=a.model)
        m = re.search(r"\[.*\]", tho_gem or "", re.S)
        cum = []
        if m:
            try:
                cum = [str(x) for x in json.loads(m.group(0))][:3]
            except (ValueError, json.JSONDecodeError):
                cum = []

        # Giữ cụm mà MỌI token của nó đều hiếm trong kho OCR.
        # Ba kết cục khác nhau, và gộp chúng lại là tự bịt mắt mình:
        #   vắng    — token không có trong kho OCR -> lọc chắc chắn ra rỗng
        #   phổ biến — token có ở > NGUONG_HIEM tài liệu -> không phân biệt được
        #   dùng    — mọi token đều có và đều hiếm
        giu, vang, pho_bien = [], [], []
        for c in cum:
            tok = [t for t in re.findall(r"[^\W_]+", bo_dau(c)) if len(t) > 1]
            if not tok:
                continue
            if any(df.get(t, 0) == 0 for t in tok):
                vang.append(c)
            elif any(df.get(t, 0) > NGUONG_HIEM for t in tok):
                pho_bien.append(c)
            else:
                giu.append((c, tok))

        moi = []
        if giu:
            mat = co.kd.apply(
                lambda x: any(all(t in x for t in tok) for _, tok in giu))
            for rid in co[mat].row_id.head(a.toi_da):
                g = master.iloc[int(rid)]
                moi.append([g.video_id, int(g.frame_idx)])

        thay, cuoi = set(), []
        for v, f in [(x[0], int(x[1])) for x in moi] + [(r[0], int(r[1])) for r in dong]:
            if (v, f) in thay:
                continue
            thay.add((v, f))
            cuoi.append((v, f))
            if len(cuoi) >= 100:
                break

        if loai == "qa":
            dap = dong[0][2] if len(dong[0]) > 2 else ""
            goi[ten] = [AnswerQA(v, f, dap) for v, f in cuoi]
        else:
            goi[ten] = [AnswerKIS(v, f) for v, f in cuoi]

        print(f"  {ten:<20} Gemini nêu {len(cum)} cụm"
              f"{' | DÙNG ' + str([x for x, _ in giu]) if giu else ''}"
              f"{' | vắng trong kho ' + str(vang) if vang else ''}"
              f"{' | quá phổ biến ' + str(pho_bien) if pho_bien else ''}"
              f" -> chèn {len(moi)} khung mới")
        time.sleep(4)          # free tier chặn theo lượt/phút

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
            for x in loi[:6]:
                print("   ", x)
            raise SystemExit(1)
        print(f"✅ Đã nén -> {z} — đạt checklist BTC")


if __name__ == "__main__":
    main()
