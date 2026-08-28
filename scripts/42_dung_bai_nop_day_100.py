r"""
42_dung_bai_nop_day_100.py — Đáp án soát tay lên đầu, TRUY HỒI đắp cho đủ 100.

    python scripts/42_dung_bai_nop_day_100.py \
        --dap-an vqa/dap_an_p2.json --bo-sung vqa/p2_kenh1 \
        --de dev/SOTUYEN2-bo-de-thi --ra vqa/nop_p2 --nen vqa/nop_p2.zip

VÌ SAO CẦN SCRIPT NÀY KHI ĐÃ CÓ `32_dung_tu_dap_an_tay.py`
==========================================================

`32_` chỉ ghi đúng những khung người đã soát — thường 1–9 dòng, rồi **bỏ trống
~95 chỗ còn lại**. Đó là vứt đi cơ hội miễn phí:

> PHẦN C mục 1: *"Luôn nộp đủ 100 câu. **Không có điểm phạt.** Câu thứ 100 vẫn
> đáng 0,2."*

Phép cộng rất đơn giản, và nó chỉ có một chiều:

* Đáp án tay **đúng** → nó ở hạng 1, các dòng đắp thêm nằm sau, **không đổi gì**.
* Đáp án tay **sai** → không đắp thì gói đó **0 điểm chắc chắn**; đắp thì còn
  cơ hội 0,2–0,6 nếu truy hồi có khung đúng đâu đó trong 100 dòng.

Không có nhánh nào mà đắp thêm làm tệ đi. Nên **luôn đắp**.

BA CHỖ DỄ SAI, ĐÃ XỬ LÝ SẴN
===========================

**1. `kf_n` KHÔNG phải `frame_idx`.** Người soát ghi theo tên file ảnh
(`095.jpg` → `kf_n=95`) vì đó là thứ họ nhìn thấy. BTC chấm theo `frame_idx`, và
`L30_V046` kf95 có `frame_idx = 6613`. Script đổi qua **cột của bảng cái**,
tuyệt đối không tính lại từ `pts_time` (lệch 1 frame — bẫy đầu tiên CLAUDE.md).

**2. Dòng trùng.** Hai `row_id` khác nhau vẫn ra cùng `(video, frame_idx)` (A5.7
— 614 keyframe trùng `frame_idx`). Đắp mà không lọc thì mất chỗ vô ích, nên
script bỏ trùng theo đúng cặp sẽ nộp.

**3. Q&A đắp thêm phải MANG THEO ĐÁP ÁN.** BTC xét `answer` theo **TỪNG dòng**,
không xét mỗi hạng 1. Dòng đắp mà để `answer` trống thì đúng khung cũng 0 điểm.
Nên dòng đắp lấy luôn chuỗi đáp án của gói.

Hệ quả có lợi: đáp án còn phân vân thì ghi **nhiều chuỗi** (`"dap_an": ["20",
"15", "15-20"]`). Mỗi chuỗi chiếm một dòng ở đầu với cùng khung; trúng chuỗi nào
cũng vẫn nằm trong top-5, tức 0,80 thay vì 1,00. Mất 0,20 để mua bảo hiểm.
⚠️ Chưa kiểm được BTC có khử dòng trùng phía họ không — đây là **giả định**.

ĐỊNH DẠNG FILE ĐÁP ÁN (JSON)
============================

    {
      "query-p2-4-kis":  {"video": "L30_V072", "kf": [9, 10]},
      "query-p2-9-qa":   {"video": "L26_V161", "kf": [57, 58, 59, 60, 61],
                          "dap_an": "Cá sòng"},
      "query-p2-23-qa":  {"video": "L25_V012", "kf": [159, 160],
                          "dap_an": ["15-20", "20", "15"]},
      "query-p2-8-trake":{"video": "L27_V011", "kf": [159, 160, 164, 166]}
    }

File này chứa nội dung suy từ đề thi → để trong `vqa/` (đã bị `.gitignore`
chặn), **không commit**.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
from nop_bai import TOI_DA_DONG, dong_goi, ghi_goi, soat_zip   # noqa: E402
from schema import AnswerKIS, AnswerQA, AnswerTRAKE    # noqa: E402


def doc_bo_sung(f: Path, loai: str):
    """CSV truy hồi -> danh sách thô. TRAKE mỗi dòng là một CHUỖI khung."""
    if not f.exists():
        return []
    ra = []
    for h in csv.reader(f.open(encoding="utf-8")):
        if not h:
            continue
        if loai == "trake":
            ra.append((h[0], [int(x) for x in h[1:] if x.strip()]))
        else:
            ra.append((h[0], int(h[1])))
    return ra


def main():
    ap = argparse.ArgumentParser(description="dap an tay + truy hoi dap du 100")
    ap.add_argument("--dap-an", required=True, type=Path)
    ap.add_argument("--bo-sung", type=Path, default=None,
                    help="thư mục CSV truy hồi để đắp thêm")
    ap.add_argument("--de", required=True, type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--ra", required=True, type=Path)
    ap.add_argument("--nen", metavar="FILE.zip")
    ap.add_argument("--khong-dap", action="store_true",
                    help="chỉ ghi đáp án tay, KHÔNG đắp (giống 32_)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    # (video_id, kf_n) -> frame_idx, tra một lần thay vì lọc lại từng dòng
    fi = {(v, int(k)): int(f) for v, k, f
          in zip(master.video_id.values, master.kf_n.values, master.frame_idx.values)}

    de = R.doc_de(a.de)
    tay = json.loads(a.dap_an.read_text("utf-8"))
    la = set(tay) - set(de)
    if la:
        raise SystemExit(f"Đáp án có gói không nằm trong đề: {sorted(la)}")

    goi, so_su_kien, canh_bao = {}, {}, []
    bo_lech = {}

    for ten in sorted(de):
        loai = R.loai_cua(ten)
        m = tay.get(ten)
        dong, da_co = [], set()

        # ---- 1. đáp án soát tay lên trước -------------------------------
        n_tay = 0
        dap_list = []
        if m:
            vid = m["video"]
            frames = []
            for k in m["kf"]:
                f = fi.get((vid, int(k)))
                if f is None:
                    canh_bao.append(f"{ten}: {vid} kf{k} không có trong bảng cái")
                else:
                    frames.append(f)
            d = m.get("dap_an", "")
            dap_list = [d] if isinstance(d, str) else list(d)

            if loai == "trake" and frames:
                n = len(R.tach_su_kien(de[ten]))
                if len(frames) != n:
                    canh_bao.append(
                        f"{ten}: đáp án tay có {len(frames)} khung nhưng đề "
                        f"tách ra {n} sự kiện — SỐ FRAME ID PHẢI KHỚP")
                dong.append(AnswerTRAKE(vid, sorted(frames)))
                so_su_kien[ten] = n
                da_co.add((vid, tuple(sorted(frames))))
            elif loai == "qa" and frames:
                # mỗi chuỗi đáp án chiếm một dòng, khung chắc nhất trước
                for dap in (dap_list or [""]):
                    for f in frames:
                        if (vid, f, dap) in da_co:
                            continue
                        da_co.add((vid, f, dap))
                        dong.append(AnswerQA(vid, f, dap))
            else:
                for f in frames:
                    if (vid, f) in da_co:
                        continue
                    da_co.add((vid, f))
                    dong.append(AnswerKIS(vid, f))
            n_tay = len(dong)

        # ---- 2. đắp bằng truy hồi cho đủ 100 ----------------------------
        if not a.khong_dap and a.bo_sung:
            dap_dap = dap_list[0] if dap_list else "không rõ"
            for v, f in doc_bo_sung(a.bo_sung / f"{ten}.csv", loai):
                if len(dong) >= TOI_DA_DONG:
                    break
                if loai == "trake":
                    # Số khung mỗi chuỗi PHẢI khớp số sự kiện của ĐỀ. Kết quả
                    # truy hồi có thể đã sinh bằng bản `tach_su_kien` cũ (A40,
                    # A44) và mang sai số khung — lúc đó `soat` KHÔNG bắt được,
                    # vì nó chỉ kiểm nhất quán nội bộ chứ không nhìn đề.
                    n = so_su_kien.setdefault(ten, len(R.tach_su_kien(de[ten])))
                    if len(f) != n:
                        bo_lech[ten] = bo_lech.get(ten, 0) + 1
                        continue
                    khoa = (v, tuple(f))
                    if khoa in da_co:
                        continue
                    da_co.add(khoa)
                    dong.append(AnswerTRAKE(v, f))
                elif loai == "qa":
                    if (v, f, dap_dap) in da_co:
                        continue
                    da_co.add((v, f, dap_dap))
                    dong.append(AnswerQA(v, f, dap_dap))
                else:
                    if (v, f) in da_co:
                        continue
                    da_co.add((v, f))
                    dong.append(AnswerKIS(v, f))

        if not dong:
            canh_bao.append(f"{ten}: KHÔNG có dòng nào — gói này chắc chắn 0 điểm")
            continue
        goi[ten] = dong
        nhan = f"tay {n_tay:>2}" if n_tay else "  —   "
        print(f"  {ten:<20} {nhan} + đắp {len(dong) - n_tay:>3} = {len(dong):>3} dòng"
              f"{'  | ' + repr(dap_list) if dap_list else ''}")

    if canh_bao:
        print("\n⚠️  CẢNH BÁO:")
        for x in canh_bao:
            print("   ", x)

    d = ghi_goi(goi, a.ra, so_su_kien)      # tự soát; có lỗi thì KHÔNG ghi gì
    print(f"\n✅ Đã ghi {d}  ({len(goi)}/{len(de)} gói)")

    if a.nen:
        z = dong_goi(d, a.nen)
        loi, canh = soat_zip(z)
        for x in canh:
            print("⚠️ ", x)
        if loi:
            print(f"\n❌ {len(loi)} LỖI — ĐỪNG NỘP:")
            for x in loi:
                print("   ", x)
        else:
            print(f"✅ {z} soát sạch")


if __name__ == "__main__":
    main()
