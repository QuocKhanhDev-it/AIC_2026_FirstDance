"""
30_xep_lai_thi_giac.py — Xếp lại top-N bằng cách cho Gemini NHÌN ẢNH THẬT.

VÌ SAO — A28 đo được trần, và chỉ đúng chỗ thủng
=================================================

Xếp lại bằng CHỮ (`28_xep_lai_bang_gemini.py`) đưa điểm 3,8 -> 5,4 rồi **dừng
hẳn**: quét sâu hơn top-50 không thêm gì, đưa thêm tiêu đề video cũng không.

Chẩn đoán của chính script nói vì sao: **8/18 gói KIS trả về danh sách rỗng** —
Gemini không tìm được bằng chứng nào trong văn bản. Đó là các truy vấn thuần thị
giác ("người phụ nữ thái cà chua bên chảo"), và OCR của một keyframe thường tả
một tin khác hẳn cảnh trong khung.

Nói cách khác: bộ xếp lại đang mù đúng ở nhóm gói mà nó cần giúp nhất. Cho nó
nhìn ảnh là cách thẳng nhất để mở nhóm đó.

VẪN GIỮ ĐÚNG QUY LUẬT ĐÃ ĐO (A28)
==================================

    XẾP LẠI trong bể kênh 1 đã chọn   -> +1,6 điểm
    THAY THẾ bằng ứng viên mới         -> -0,4 điểm

Script này **chỉ xếp lại**. Không ứng viên nào bị bỏ, không khung nào mới được
thêm vào. Kịch bản xấu nhất là xáo nhầm thứ tự trong top-N.

CHỈ XẾP LẠI KHUNG CÓ ẢNH TRÊN MÁY NÀY
======================================

`kf_path` phụ thuộc máy (A5.5): máy đang chạy chỉ có ảnh của L21/L22/L24/L27/L30
— 36.506/177.321 dòng. Đo trên bài nộp 5,4: **537/900 khung** trong top-50 của
18 gói KIS có ảnh (60%), nhưng lệch rất mạnh — 6 gói đủ 50/50, có gói chỉ 2/50.

Nên: khung KHÔNG có ảnh **giữ nguyên vị trí tương đối**, chỉ những khung nhìn
được mới tham gia xếp lại. Gói nào ít ảnh thì script gần như không đụng tới —
đó là hành vi đúng, không phải hỏng.

    python scripts/30_xep_lai_thi_giac.py --nguon firstdance6.zip \\
        --de dev/THUNGHIEM-bo-de-thi --ra submission_v11 --nen firstdance10.zip
"""

import argparse
import csv
import io
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
from nop_bai import dong_goi, ghi_goi, soat_zip        # noqa: E402
from schema import AnswerKIS, AnswerQA, AnswerTRAKE    # noqa: E402
from tra_loi_ocr import (MODEL_GEMINI, _goi_gemini_anh,  # noqa: E402
                         thu_nho)

NHAC = """Bạn đang xếp hạng lại kết quả tìm kiếm video theo một mô tả tiếng Việt.

MÔ TẢ CẦN TÌM:
{truy_van}

Dưới đây là {n} khung hình, đánh số theo đúng thứ tự xuất hiện của ảnh: ảnh thứ
nhất là khung [1], ảnh thứ hai là khung [2], v.v.

Nhiệm vụ: liệt kê số thứ tự những khung KHỚP với mô tả, khung khớp nhất trước.

Quy tắc bắt buộc:
- Chỉ nêu khung thật sự khớp về CẢNH được mô tả (chủ thể, hành động, bối cảnh).
- Không khung nào khớp -> trả về [].
- Đừng nêu quá {toi_da} khung.
- Trả lời ĐÚNG một dòng JSON, không giải thích. Ví dụ: [3, 7, 1]"""


def doc_zip(z: Path) -> dict:
    ra = {}
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if not i.is_dir() and i.filename.lower().endswith(".csv"):
                ra[Path(i.filename).stem] = [
                    r for r in csv.reader(
                        io.StringIO(f.read(i.filename).decode("utf-8"))) if r]
    return ra


def duong_dan_anh(master, video_id: str, frame_idx: int):
    g = master[(master.video_id == video_id)
               & (master.frame_idx.astype(int) == int(frame_idx))]
    if g.empty:
        return None
    p = g.kf_path.iloc[0]
    return p if isinstance(p, str) and p and Path(p).exists() else None


def xep_lai(dong: list, truy_van: str, master, top: int, so_anh: int,
            model: str) -> tuple[list, list, int]:
    """Trả `(thứ tự mới, vị trí được đẩy lên, số ảnh đã gửi)`."""
    dau = dong[:top]
    # vị trí (1-based trong `dau`) -> bytes ảnh
    nhin_duoc, anh = [], []
    for i, r in enumerate(dau, 1):
        if len(anh) >= so_anh:
            break
        p = duong_dan_anh(master, r[0], r[1])
        if not p:
            continue
        b = thu_nho(p)
        if b:
            nhin_duoc.append(i)
            anh.append(b)
    if len(anh) < 2:
        return dong, [], len(anh)

    tho = _goi_gemini_anh(
        NHAC.format(truy_van=truy_van, n=len(anh), toi_da=max(3, len(anh) // 3)),
        anh, model=model)
    m = re.search(r"\[[\d,\s]*\]", tho or "")
    if not m:
        return dong, [], len(anh)
    try:
        chon = [int(x) for x in json.loads(m.group(0))]
    except (ValueError, json.JSONDecodeError):
        return dong, [], len(anh)

    # Gemini đánh số theo THỨ TỰ ẢNH ĐÃ GỬI, không phải theo hạng gốc — ánh xạ
    # ngược lại, nếu không sẽ đẩy nhầm khung.
    vi_tri = [nhin_duoc[i - 1] for i in dict.fromkeys(chon)
              if 1 <= i <= len(nhin_duoc)]
    if not vi_tri:
        return dong, [], len(anh)

    len_ = [dau[i - 1] for i in vi_tri]
    con_lai = [r for i, r in enumerate(dau, 1) if i not in set(vi_tri)]
    return len_ + con_lai + dong[top:], vi_tri, len(anh)


def main():
    ap = argparse.ArgumentParser(description="xep lai top-N bang cach NHIN anh")
    ap.add_argument("--nguon", required=True, type=Path)
    ap.add_argument("--de", required=True, type=Path)
    ap.add_argument("--ra", default=Path("submission_thigiac"), type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--top", type=int, default=50,
                    help="phạm vi xếp lại (A28 chốt 50)")
    ap.add_argument("--so-anh", type=int, default=20,
                    help="số ảnh gửi mỗi lượt gọi — càng nhiều càng chậm/tốn")
    ap.add_argument("--model", default=MODEL_GEMINI)
    ap.add_argument("--nen", metavar="FILE.zip")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    de = R.doc_de(a.de)
    tho = doc_zip(a.nguon)
    print(f"{a.nguon}: {len(tho)} gói | nhìn tối đa {a.so_anh} ảnh/gói "
          f"trong top-{a.top}\n")

    goi, so_su_kien = {}, {}
    for ten, dong in sorted(tho.items()):
        loai = ten.rsplit("-", 1)[-1]
        if loai == "trake":
            so_su_kien[ten] = len(dong[0]) - 1
            goi[ten] = [AnswerTRAKE(r[0], [int(x) for x in r[1:]]) for r in dong]
            continue
        if loai == "qa":
            goi[ten] = [AnswerQA(r[0], int(r[1]), r[2] if len(r) > 2 else "")
                        for r in dong]
            continue

        moi, chon, n_anh = xep_lai(dong, de.get(ten, ""), master, a.top,
                                   a.so_anh, a.model)
        goi[ten] = [AnswerKIS(r[0], int(r[1])) for r in moi]
        print(f"  {ten:<20} nhìn {n_anh:>2} ảnh -> đẩy lên {len(chon):>2}"
              f"{'  ' + str(chon[:5]) if chon else '  (không khung nào khớp)'}")
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
