"""
38_soat_cau_dev_moi.py — Soát một file câu dev MỚI trước khi gộp vào tập dev.

    python scripts/38_soat_cau_dev_moi.py dev/tap_dev_thanh_vien/tap_dev_X.jsonl

VÌ SAO CẦN SOÁT TRƯỚC KHI GỘP
==============================

Một câu dev sai **không crash gì cả** — nó chỉ lặng lẽ làm mọi phép đo sau đó
lệch đi. Ba kiểu sai đã cắn thật, và script này bắt cả ba:

**1. RÒ VĂN BẢN — kiểu nguy hiểm nhất (A21).** Nếu người soạn chép chữ đang
chạy trên màn hình vào câu hỏi, kênh OCR sẽ "tìm ra" đáp án mà không thật sự
nhìn thấy gì. A21 đo được mức tăng **ảo** 0,400 → 0,840 đúng vì chuyện này.

Triệu chứng đo được: **kênh OCR xếp đáp án hạng rất cao**. Một câu tả hình ảnh
thuần thì OCR gần như không tìm ra gì. Ngưỡng ở đây là **hạng ≤ 10** — không
phải bằng chứng, mà là **cờ để người soạn đọc lại câu đó**.

**2. PHÂN BỐ LỆCH — lý do tập dev đã mù SÁU lần** (A19/A20/A31/A34/A37/A41).
Câu tự soạn cũ: trung vị **22 từ / 1,39 mệnh đề**. Đề thật: **60 từ / 2,33**.
Đo trên câu ngắn rồi suy ra cho câu dài là chỗ hỏng đã lặp lại nhiều nhất.

**3. ĐÁP ÁN KHÔNG TRA ĐƯỢC.** `row_id` trỏ sang video khác, hoặc trỏ vào khung
máy này không có ảnh (79,4% kho — A5.5) nên người soạn không thể đã nhìn thấy nó.
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
from bm25 import KenhVanBan                            # noqa: E402

# Đích đo được từ 24 gói KIS/QA của đề sơ tuyển đợt 1.
DICH_TU, DICH_MD = 60, 2.33
CU_TU, CU_MD = 22, 1.39


def main():
    ap = argparse.ArgumentParser(description="soat cau dev moi truoc khi gop")
    ap.add_argument("file", type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--nguong-ocr", type=int, default=10,
                    help="OCR xếp đáp án trong top này -> cờ nghi rò văn bản")
    a = ap.parse_args()

    cau = [json.loads(l) for l in a.file.read_text("utf-8").splitlines() if l.strip()]
    master = pd.read_parquet(a.index / "master.parquet")
    vid = master.video_id.values
    kfp = master.kf_path.values

    print(f"{len(cau)} câu trong {a.file.name}\n")

    # ---- 1. phân bố -----------------------------------------------------
    tu = [len(c["cau_hoi"].split()) for c in cau]
    md = [len(R.tach_truy_van(c["cau_hoi"])) for c in cau]
    n2 = sum(1 for x in md if x >= 2)
    print("PHÂN BỐ (đích = đề thật)")
    print(f"  từ/câu   : trung vị {st.median(tu):5.0f}   "
          f"đích {DICH_TU} | tự soạn cũ {CU_TU}")
    print(f"  mệnh đề  : trung bình {st.mean(md):5.2f}   "
          f"đích {DICH_MD} | tự soạn cũ {CU_MD}")
    print(f"  >=2 mệnh đề: {n2}/{len(cau)}   đích ~75%")
    if st.median(tu) < 40:
        print("  ⚠️  NGẮN hơn đề thật nhiều — đây đúng chỗ tập dev đã mù 6 lần")
    print()

    # ---- 2. đáp án tra được ---------------------------------------------
    loi = 0
    for c in cau:
        rs = [r[0] if isinstance(r, list) else r for r in c["row_id_dung"]]
        if not rs:
            print(f"  ❌ {c['id']}: không có row_id nào"); loi += 1; continue
        vs = {str(vid[r]) for r in rs}
        if len(vs) > 1:
            print(f"  ❌ {c['id']}: row_id trải nhiều video {sorted(vs)}"); loi += 1
        thieu_anh = [r for r in rs if pd.isna(kfp[r])]
        if thieu_anh:
            print(f"  ⚠️  {c['id']}: {len(thieu_anh)}/{len(rs)} khung KHÔNG có ảnh "
                  f"ở máy này — người soạn nhìn thấy nó bằng cách nào?")
    print(f"ĐÁP ÁN: {'✅ tra được hết' if not loi else f'❌ {loi} câu hỏng'}\n")

    # ---- 3. rò văn bản ---------------------------------------------------
    p = a.index / "ocr_asr.parquet"
    if not p.exists():
        print("(không có ocr_asr.parquet — bỏ qua phép soát rò văn bản)")
        return
    print("RÒ VĂN BẢN — kênh OCR xếp đáp án ở hạng nào?")
    print("  (câu tả hình ảnh thuần thì OCR gần như không tìm ra: hạng '—')")
    ocr = KenhVanBan.tu_bang_khung(master, pd.read_parquet(p),
                                   cot="text", ten="ocr_asr")
    nghi = []
    for c in cau:
        rs = {r[0] if isinstance(r, list) else r for r in c["row_id_dung"]}
        kq = ocr.tim(R.tach_truy_van(c["cau_hoi"]), k=100)
        h = next((i + 1 for i, x in enumerate(kq) if x.row_id in rs), None)
        co = ""
        if h and h <= a.nguong_ocr:
            co = "  ⚠️ NGHI RÒ — đọc lại câu này"
            nghi.append(c["id"])
        print(f"  {c['id']:14} {(str(h) if h else '—'):>5}{co}")
    print()
    if nghi:
        print(f"⚠️  {len(nghi)} câu nghi rò: {', '.join(nghi)}")
        print("   Không phải bằng chứng — nhưng hãy mở lại contact sheet và hỏi:")
        print("   cụm từ đó mình lấy từ HÌNH hay từ CHỮ CHẠY dưới màn hình?")
    else:
        print("✅ Không câu nào bị OCR tìm ra sớm — không thấy dấu hiệu rò.")


if __name__ == "__main__":
    main()
