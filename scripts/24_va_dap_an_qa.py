"""
24_va_dap_an_qa.py — Vá đáp án Q&A ĐÃ XÁC MINH vào một bài nộp đã có.

VÌ SAO CẦN — A24 đo được điều này trên bài nộp thật:

    3/24 gói của bộ đề mẫu là Q&A, và **cả bốn lượt nộp đều chắc chắn 0 điểm**
    ở ba gói đó. Không phải vì truy hồi kém, mà vì `answer` sai: lượt 3,8 nộp
    `5`, `2`, `10` trong khi đề hỏi TÊN MỘT XÃ, HAI CÂU THƠ và TÊN MÓN ĂN.

BTC chấm Q&A theo `khung đúng VÀ answer đúng` (PHẦN C mục 4). Nên khi người ta
đã **tìm ra đáp án bằng mắt**, thứ cần làm là đưa cả hai vào cùng một dòng: đặt
khung chứa đáp án lên hạng 1, và gắn chuỗi đáp án cho mọi dòng của gói đó.

    python scripts/24_va_dap_an_qa.py --nguon nop_cu.zip --ra submission_v3 \\
        --va "query-p1-15-qa|L30_V072|1745|Giang Ly" \\
        --lan-can 10

GIỮ NGUYÊN MỌI GÓI KHÁC
=======================

Script này **không truy hồi lại gì cả**. Nó chép nguyên các gói KIS/TRAKE từ
bài nộp nguồn — thứ đã ăn điểm cao nhất của đội — và chỉ sửa đúng những gói Q&A
được chỉ định. Đổi một thứ mỗi lần, đúng kỷ luật đo của repo: nếu điểm lên thì
biết chắc phần lên đến từ đâu.

VÌ SAO THÊM KHUNG LÂN CẬN
=========================

Đáp án chuẩn của BTC là một KHOẢNG rộng 4 giây–5 phút (A9), không phải một
frame. Khung ta xác minh nằm giữa khoảng đó, nhưng ta không biết khoảng rộng bao
nhiêu và bắt đầu từ đâu. Rải thêm vài khung liền kề **trong cùng video** là bảo
hiểm rẻ: chúng chiếm những chỗ mà `bu_cho_du` vốn điền bằng khung ngẫu nhiên.

⚠️ Nhưng để chúng SAU khung đã xác minh. R@1 chiếm 1/5 tổng điểm — đẩy khung
chắc chắn đúng xuống hạng 2 để nhường chỗ cho một phỏng đoán là tự bỏ điểm.
"""

import argparse
import csv
import io
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from nop_bai import (TOI_DA_DONG, dong_goi, ghi_goi,           # noqa: E402
                     soat_zip)
from schema import AnswerKIS, AnswerQA, AnswerTRAKE            # noqa: E402


def doc_zip(z: Path) -> dict[str, list[list[str]]]:
    """`{tên gói: [các dòng CSV thô]}`."""
    ra = {}
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if i.is_dir() or not i.filename.lower().endswith(".csv"):
                continue
            ten = Path(i.filename).stem
            ra[ten] = [r for r in csv.reader(
                io.StringIO(f.read(i.filename).decode("utf-8"))) if r]
    return ra


def khung_lan_can(master: pd.DataFrame, video_id: str, frame_idx: int,
                  so_buoc: int) -> list[int]:
    """`frame_idx` của các keyframe liền kề trong CÙNG video, gần nhất trước.

    Sắp theo khoảng cách tăng dần tới khung neo, nên khung càng gần đáp án càng
    đứng trên — cùng lý do với thứ tự ưu tiên ở `mui_nhon_1.khung_ngu_canh`.
    """
    g = master[master.video_id == video_id]
    if g.empty:
        raise SystemExit(f"Không có video {video_id} trong bảng cái")
    g = g.assign(cach=(g.frame_idx.astype(int) - frame_idx).abs())
    return [int(x) for x in g.sort_values("cach").frame_idx.head(so_buoc + 1)]


def va(goi: dict, master: pd.DataFrame, cac_va: list, so_lan_can: int,
       k: int = TOI_DA_DONG) -> dict:
    """Trả về gói mới: Q&A được vá, mọi gói khác giữ nguyên."""
    ra = dict(goi)
    for ten, video_id, frame_idx, dap_an in cac_va:
        if ten not in goi:
            raise SystemExit(f"Bài nộp nguồn không có gói {ten!r}. "
                             f"Có: {', '.join(sorted(goi))}")
        if not dap_an.strip():
            raise SystemExit(f"{ten}: đáp án rỗng — `nop_bai.soat` sẽ chặn")

        # 1. khung ĐÃ XÁC MINH đứng đầu, 2. lân cận, 3. phần còn lại của bài cũ
        thu_tu = [(video_id, int(frame_idx))]
        for f in khung_lan_can(master, video_id, int(frame_idx), so_lan_can):
            thu_tu.append((video_id, f))
        for cu in goi[ten]:            # phần còn lại của bài nộp nguồn
            thu_tu.append((cu.video_id, int(cu.frame_idx)))

        thay, dong_moi = set(), []
        for v, f in thu_tu:
            if (v, f) in thay:
                continue
            thay.add((v, f))
            dong_moi.append(AnswerQA(v, f, dap_an))
            if len(dong_moi) >= k:
                break
        ra[ten] = dong_moi
        print(f"  {ten}: hạng 1 = {video_id} frame {frame_idx} | "
              f"answer {dap_an!r} ({len(dap_an)} ký tự) | {len(dong_moi)} dòng")
    return ra


def _thanh_dap_an(ten: str, dong: list[list[str]]) -> list:
    """Dòng CSV thô -> đối tượng đáp án, theo hậu tố tên gói."""
    loai = ten.rsplit("-", 1)[-1]
    if loai == "kis":
        return [AnswerKIS(r[0], int(r[1])) for r in dong]
    if loai == "qa":
        return [AnswerQA(r[0], int(r[1]), r[2] if len(r) > 2 else "") for r in dong]
    return [AnswerTRAKE(r[0], [int(x) for x in r[1:]]) for r in dong]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--nguon", required=True, type=Path,
                    help="file .zip đã nộp, dùng làm nền")
    ap.add_argument("--ra", default=Path("submission_va"), type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--va", action="append", default=[], metavar="CHUOI",
                    help='"tên gói|video_id|frame_idx|đáp án" — lặp lại được')
    ap.add_argument("--lan-can", type=int, default=10,
                    help="số khung liền kề chèn sau khung đã xác minh")
    ap.add_argument("--nen", metavar="FILE.zip")
    a = ap.parse_args()

    if not a.va:
        raise SystemExit("Chưa có --va nào. Không có gì để vá.")
    cac_va = []
    for s in a.va:
        phan = s.split("|")
        if len(phan) != 4:
            raise SystemExit(f"--va sai định dạng: {s!r}\n"
                             f'   Đúng: "tên gói|video_id|frame_idx|đáp án"')
        cac_va.append((phan[0].strip(), phan[1].strip(), phan[2].strip(), phan[3]))

    master = pd.read_parquet(a.index / "master.parquet")
    tho = doc_zip(a.nguon)
    print(f"{a.nguon}: {len(tho)} gói\n")

    goi = {ten: _thanh_dap_an(ten, dong) for ten, dong in tho.items()}
    goi = va(goi, master, cac_va, a.lan_can)

    if Path(a.ra).exists():
        shutil.rmtree(a.ra)

    # Số sự kiện TRAKE lấy từ CHÍNH bài nộp nguồn — bài đó BTC đã nhận và đã
    # chấm, nên số đó là số đúng. Tách lại từ đề ở đây là mở đường cho một lỗi
    # mới ở chỗ vốn đang chạy tốt.
    so_su_kien = {ten: len(dong[0]) - 1 for ten, dong in tho.items()
                  if ten.rsplit("-", 1)[-1] == "trake"}
    d = ghi_goi(goi, a.ra, so_su_kien)      # tự soát; có lỗi thì KHÔNG ghi gì
    print(f"\n✅ Đã ghi {d}")

    if a.nen:
        z = dong_goi(d, a.nen)
        loi, canh = soat_zip(z)
        for x in canh:
            print("⚠️ ", x)
        if loi:
            print(f"\n❌ {len(loi)} LỖI — ĐỪNG NỘP:")
            for x in loi:
                print("   ", x)
            raise SystemExit(1)
        print(f"✅ Đã nén -> {z} — đạt checklist BTC")


if __name__ == "__main__":
    main()
