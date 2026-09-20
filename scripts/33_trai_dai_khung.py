"""
33_trai_dai_khung.py — Trải khung đã soát tay ra cho đủ 100 dòng mỗi gói.

Bài nộp dựng từ đáp án soát tay chỉ có **1–9 dòng** mỗi gói. Luật BTC: **không
có điểm phạt**, và *"dòng thứ 100 vẫn đáng 0,2 điểm"* (PHẦN C mục 1). Nghĩa là
91–99 chỗ trống kia là **bảo hiểm miễn phí**:

* đáp án tay trúng  -> mấy dòng thêm vào vô hại, hạng 1 vẫn là hạng 1
* đáp án tay trượt  -> vẫn còn cơ hội gỡ ở R@5 / R@20 / R@50 / R@100

    python scripts/33_trai_dai_khung.py --nguon firstdancesotuyen1.zip \\
        --ra sub_traidai --nen firstdancesotuyen1traidai.zip

HAI NGUYÊN TẮC, VÀ CẢ HAI ĐỀU CÓ LÝ DO
=======================================

**1. Khung đã soát tay GIỮ NGUYÊN thứ hạng đầu.** R@1 chiếm 1/5 tổng điểm; đẩy
một khung người đã mở ảnh xác nhận xuống dưới một khung máy đoán là tự bỏ điểm.
Script chỉ THÊM vào sau, không bao giờ chen lên trước.

**2. Chỉ trải TRONG CÙNG video mà nhóm đã chốt.** Nhóm đã xem ảnh và xác nhận
video — đó là phần chắc nhất của đáp án. Nếu video đúng mà khoảnh khắc lệch thì
trải trong video sẽ vớt được; nếu video sai thì thêm video khác cũng chỉ là đoán
mò, mà đoán mò đã đo được là **làm tệ đi** (A28: thay ứng viên mới = −0,4 điểm).

Thứ tự thêm: **gần khung đã soát nhất trước**. Đáp án chuẩn của BTC là một cửa
sổ rộng 4 giây–5 phút (A9), nên khung liền kề có xác suất rơi trúng cửa sổ cao
hơn hẳn khung ở đầu kia video.

TRAKE TRẢI KHÁC HẲN
===================

Mỗi dòng TRAKE là một BỘ N khung, không phải một khung. Trải bằng cách **dịch
cả chuỗi** và cho từng sự kiện nhích quanh vị trí đã chốt, luôn giữ **tăng dần
theo thời gian** — BTC đòi vậy và `nop_bai.soat` sẽ chặn nếu sai.
"""

import argparse
import csv
import io
import itertools
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from nop_bai import (TOI_DA_DONG, dong_goi, ghi_goi,     # noqa: E402
                     soat_zip)
from schema import AnswerKIS, AnswerQA, AnswerTRAKE      # noqa: E402


def doc_zip(z: Path) -> dict:
    ra = {}
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if not i.is_dir() and i.filename.lower().endswith(".csv"):
                ra[Path(i.filename).stem] = [
                    r for r in csv.reader(
                        io.StringIO(f.read(i.filename).decode("utf-8"))) if r]
    return ra


def trai_mot_khung(master, video_id: str, da_co: list, k: int) -> list:
    """`frame_idx` của gói sau khi trải: đã soát trước, lân cận sau."""
    g = master[master.video_id == video_id].sort_values("frame_idx")
    # ⚠️ BỎ TRÙNG `frame_idx`. A5.7: 614 keyframe trong kho trùng hệt frame_idx
    # với dòng liền trước, nên duyệt thẳng bảng cái sẽ sinh hai dòng nộp y hệt
    # nhau — `nop_bai.soat` chặn cả gói, và đúng ra thì nó phí một trong 100 chỗ.
    tat_ca = list(dict.fromkeys(int(x) for x in g.frame_idx))
    da_co = list(dict.fromkeys(int(x) for x in da_co))
    con_lai = [f for f in tat_ca if f not in set(da_co)]
    # gần khung đã soát nhất đứng trước
    con_lai.sort(key=lambda f: min(abs(f - x) for x in da_co))
    return (da_co + con_lai)[:k]


def trai_trake(master, video_id: str, chuoi: list, k: int) -> list:
    """Bộ N khung -> nhiều bộ, dịch quanh vị trí đã chốt, luôn tăng dần."""
    g = master[master.video_id == video_id].sort_values("frame_idx")
    tat_ca = [int(x) for x in g.frame_idx]
    vi_tri = {f: i for i, f in enumerate(tat_ca)}
    goc = [vi_tri.get(f) for f in chuoi]
    if any(v is None for v in goc):
        return [chuoi]

    ra, thay = [list(chuoi)], {tuple(chuoi)}
    # bán kính nới dần: bộ gần bộ gốc nhất được thêm trước
    for r in range(1, 40):
        for lech in itertools.product(range(-r, r + 1), repeat=len(goc)):
            if max(abs(x) for x in lech) != r:      # chỉ lấy vành ngoài
                continue
            idx = [a + b for a, b in zip(goc, lech)]
            if any(i < 0 or i >= len(tat_ca) for i in idx):
                continue
            bo = [tat_ca[i] for i in idx]
            if any(b <= a for a, b in zip(bo, bo[1:])):   # phải TĂNG THẬT
                continue
            t = tuple(bo)
            if t in thay:
                continue
            thay.add(t)
            ra.append(bo)
            if len(ra) >= k:
                return ra
    return ra


def main():
    ap = argparse.ArgumentParser(description="trai khung cho du 100 dong/goi")
    ap.add_argument("--nguon", required=True, type=Path)
    ap.add_argument("--ra", default=Path("sub_traidai"), type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--k", type=int, default=TOI_DA_DONG)
    ap.add_argument("--nen", metavar="FILE.zip")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    tho = doc_zip(a.nguon)
    goi, so_su_kien = {}, {}

    for ten, dong in sorted(tho.items()):
        loai = ten.rsplit("-", 1)[-1]
        vid = dong[0][0]
        cu = len(dong)

        if loai == "trake":
            chuoi = [int(x) for x in dong[0][1:]]
            bo = trai_trake(master, vid, chuoi, a.k)
            goi[ten] = [AnswerTRAKE(vid, x) for x in bo]
            so_su_kien[ten] = len(chuoi)
        else:
            da_co = [int(r[1]) for r in dong]
            frames = trai_mot_khung(master, vid, da_co, a.k)
            if loai == "qa":
                dap = dong[0][2] if len(dong[0]) > 2 else ""
                goi[ten] = [AnswerQA(vid, f, dap) for f in frames]
            else:
                goi[ten] = [AnswerKIS(vid, f) for f in frames]

        print(f"  {ten:<20} {vid:<10} {cu:>2} -> {len(goi[ten]):>3} dòng")

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
            print(f"\n❌ {len(loi)} LỖI — ĐỪNG NỘP:")
            for x in loi[:8]:
                print("   ", x)
            raise SystemExit(1)
        print(f"✅ Đã nén -> {z} — đạt checklist BTC")


if __name__ == "__main__":
    main()
