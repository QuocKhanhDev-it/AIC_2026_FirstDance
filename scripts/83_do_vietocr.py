"""
83_do_vietocr.py — OCR mới có đọc ra đáp án Q&A ĐÚNG DẤU không?

    python scripts/83_do_vietocr.py --moi index/ocr_vietocr.jsonl

Đọc file OCR do Kaggle sinh (xem `notebooks/kaggle_vietocr.md`) rồi trả lời ba
câu, mỗi câu một dòng trong bảng:

  1. **CÓ DẤU** — bao nhiêu % văn bản mới mang dấu tiếng Việt? OCR hiện tại chỉ
     31%. Đây là thứ quyết định rào cản 1 của A68 có được vá hay không.
  2. **KHỚP ĐÚNG DẤU** — đáp án vàng xuất hiện y nguyên trong OCR mới ở bao
     nhiêu câu? Hiện 0/13 (7/13 chỉ khớp khi bỏ dấu).
  3. **TỐC ĐỘ** — s/ảnh, suy ra giờ cho cả kho.

⚠️ NGƯỠNG ĐẶT TRƯỚC KHI XEM SỐ. OCR mới phải đưa "khớp đúng dấu" lên **ít nhất
4/13** thì mới đáng chạy lại cả kho — dưới mức đó thì phần đáp án Q&A vẫn phải
trông vào VLM, và OCR mới chỉ còn giá trị cho kênh 3 (mà BM25 đã có nhánh không
dấu nên lợi ích nhỏ).
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402

KHO = 177_321


def co_dau(s) -> bool:
    return any(unicodedata.category(c) == "Mn"
               for c in unicodedata.normalize("NFD", str(s)))


def bo_dau(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--moi", default=GOC / "index" / "ocr_vietocr.jsonl",
                    type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl", GOC / "dev" / "tap_de_thi_thu.jsonl"])
    a = ap.parse_args()

    if not a.moi.exists():
        raise SystemExit(
            f"Chưa có {a.moi}.\n"
            f"  1. python scripts/82_xuat_khung_thu_ocr.py\n"
            f"  2. chạy notebooks/kaggle_vietocr.md trên Kaggle\n"
            f"  3. tải ocr_vietocr.jsonl về index/")

    moi = {}
    giay = None
    for l in a.moi.read_text("utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("_meta"):
            giay = d.get("giay_moi_anh")
            continue
        moi[int(d["row_id"])] = d.get("text", "")

    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    cu = dict(zip(bang.row_id.astype(int), bang.ocr_text.fillna("")))

    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "QA" and c.dap_an]
    rid_qa = sorted({r for c in cau for r in c.row_id_dung})

    print(f"OCR mới: {len(moi):,} khung | {len(cau)} câu Q&A, "
          f"{len(rid_qa)} khung đáp án\n")

    print(f"{'':<22}{'CŨ':>10}{'MỚI':>10}")
    print("-" * 42)
    for ten, ids in (("mọi khung đo được", list(moi)),
                     ("riêng khung đáp án", rid_qa)):
        c_cu = sum(1 for r in ids if co_dau(cu.get(r, "")))
        c_moi = sum(1 for r in ids if co_dau(moi.get(r, "")))
        n = len(ids)
        print(f"{'có dấu — ' + ten:<22}{c_cu / n * 100:>9.0f}%"
              f"{c_moi / n * 100:>9.0f}%")
    for ten, f in (("dài trung bình (ký tự)",
                    lambda d, ids: sum(len(str(d.get(r, ""))) for r in ids) / len(ids)),):
        print(f"{ten:<22}{f(cu, rid_qa):>10.0f}{f(moi, rid_qa):>10.0f}")

    print(f"\n{'câu':<16}{'đáp án':<18}{'CŨ':>12}{'MỚI':>12}")
    print("-" * 60)
    dem = {"cu_dau": 0, "moi_dau": 0, "cu_bo": 0, "moi_bo": 0}
    for c in cau:
        vang = c.dap_an.strip()
        t_cu = " ".join(str(cu.get(r, "")) for r in c.row_id_dung)
        t_moi = " ".join(str(moi.get(r, "")) for r in c.row_id_dung)
        kq = []
        for t, k in ((t_cu, "cu"), (t_moi, "moi")):
            dau = vang.lower() in t.lower()
            bo = bo_dau(vang) in bo_dau(t)
            dem[f"{k}_dau"] += dau
            dem[f"{k}_bo"] += bo
            kq.append("✅ đúng dấu" if dau else ("~ bỏ dấu" if bo else "—"))
        print(f"{c.id:<16}{vang[:16]:<18}{kq[0]:>12}{kq[1]:>12}")

    n = len(cau)
    print(f"\n{'KHỚP ĐÚNG DẤU':<22}{dem['cu_dau']:>6}/{n}{dem['moi_dau']:>10}/{n}"
          f"   <- con số quyết định")
    print(f"{'khớp khi bỏ dấu':<22}{dem['cu_bo']:>6}/{n}{dem['moi_bo']:>10}/{n}")

    if giay:
        print(f"\ntốc độ: {giay:.2f} s/ảnh -> cả kho {giay * KHO / 3600:.1f} giờ"
              f" | chia 12 phần: {giay * KHO / 12 / 3600:.1f} giờ/phần")
    print("\n⚠️ Ngưỡng đặt trước: 'khớp đúng dấu' phải >= 4/13 thì mới đáng "
          "chạy lại cả kho.")


if __name__ == "__main__":
    main()
