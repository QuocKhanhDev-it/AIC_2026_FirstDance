"""
20_do_run.py — Chấm THỨ THẬT SỰ ĐƯỢC NỘP, không phải thứ kênh trả về.

Mọi phép đo từ trước tới nay (A10–A17) chấm **kênh**: lấy `list[Candidate]` rồi
tìm xem đáp án nằm ở hạng mấy. Nhưng bài nộp còn đi qua một tầng nữa —
`run.dung_trake()`, `nop_bai.tu_ung_vien()`, cắt 100 dòng, ép thứ tự thời gian
— và **tầng đó chưa ai đo**. Nếu nó làm mất điểm thì hiện không ai biết: đúng
loại hỏng im lặng mà cả repo này dựng để chặn.

    python scripts/20_do_run.py --matrix clip.npy         # kiểm nối dây, máy yếu
    python scripts/20_do_run.py --matrix clip_siglip2.npy # số thật, máy >= 16 GB

HAI CHỖ PHÉP ĐO NÀY KHÁC MỌI PHÉP ĐO TRƯỚC
==========================================

**1. Đi qua `(video_id, frame_idx)` rồi mới quay lại `row_id`.** Bài nộp không
mang `row_id` — nó mang tên video và số frame. Nên script dựng đáp án đúng như
lúc nộp, rồi tra ngược. Vòng quay này bắt được lỗi mà chấm trên `row_id` không
bao giờ thấy: A5.7 đo được **614 keyframe trùng hệt `frame_idx`** với dòng liền
trước, nên ánh xạ ngược **không phải song ánh**.

**2. TRAKE chấm theo `diem_trake_bai_nop()`, không phải `diem_trake()`.** Cái
sau hỏi *"kênh có tìm ra các sự kiện không"*; cái trước hỏi *"nộp bộ N khung
này thì được mấy điểm"* — vị trí i chỉ được so với sự kiện i. Kênh tìm đủ ba sự
kiện mà lắp sai vị trí thì BTC cho 0.
"""

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                                  # noqa: E402
from cham_diem import (MOC_DUNG_SAI, diem_cau, diem_trake_bai_nop,  # noqa: E402
                       no_cua_so, _hang, _dung_dap_an)
from nop_bai import TOI_DA_DONG, tu_ung_vien                    # noqa: E402
from schema import AnswerTRAKE                                  # noqa: E402


def _nap_run():
    s = importlib.util.spec_from_file_location("run_mod", GOC / "src" / "run.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def bang_tra_nguoc(master) -> dict:
    """`(video_id, frame_idx) -> [row_id, ...]`.

    Danh sách chứ không phải một giá trị: A5.7 đo được 614 keyframe trùng hệt
    `frame_idx` với dòng liền trước. Nộp một `frame_idx` như thế là trúng BẤT
    KỲ dòng nào trong nhóm — nên phải giữ cả nhóm rồi mới đối chiếu.
    """
    d = defaultdict(list)
    for r, v, f in zip(master.row_id.values, master.video_id.values,
                       master.frame_idx.values):
        d[(v, int(f))].append(int(r))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--matrix", default="clip.npy")
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--k", type=int, default=TOI_DA_DONG)
    ap.add_argument("--dung-sai", type=float, nargs="+", default=list(MOC_DUNG_SAI))
    ap.add_argument("--dap-an", default="hoan-hao",
                    choices=("hoan-hao", "trong", "vlm"),
                    help="hoan-hao = dung dap an dung cua tap dev -> TRAN TREN "
                         "(gia dinh VLM luon dung); trong = SAN DUOI (chua co "
                         "VLM); vlm = goi Qwen2.5-VL that")
    ap.add_argument("--kenh", default="anh", choices=("anh", "objects"),
                    help="objects = khong can model lon, dung de KIEM NOI DAY "
                         "tren may yeu; anh = so that")
    a = ap.parse_args()

    R = _nap_run()
    master = pd.read_parquet(a.index / "master.parquet")
    tra = bang_tra_nguoc(master)
    cau = tap_dev.doc(a.file)
    print(f"{len(cau)} câu | ma trận {a.matrix} | k={a.k}\n")

    # Chạy kênh MỘT lần cho mọi câu, dùng đúng hàm của run.py
    de = {}
    for c in cau:
        ten = f"query-{len(de)+1}-{c.loai.lower()}"
        de[ten] = c.cau_hoi
    ten_theo_cau = dict(zip(de, cau))

    if a.kenh == "objects":
        # Duong chay KHONG CAN MODEL — chi de kiem tang lap rap. Kenh objects
        # duoc 0,0412 (A14) nen con so tuyet doi khong noi len gi ve he thong;
        # cai can nhin la duong ong co chay tron ven khong.
        s16 = importlib.util.spec_from_file_location(
            "r16", GOC / "scripts" / "16_do_rrf.py")
        r16 = importlib.util.module_from_spec(s16)
        s16.loader.exec_module(r16)
        k4 = r16.KenhObjects(a.index, master)
        kq1 = {}
        for ten, nd in de.items():
            if ten.endswith("-trake"):
                kq1[ten] = [k4.tim(x, k=a.k) for x in R.tach_su_kien(nd)]
            else:
                kq1[ten] = k4.tim(nd, k=a.k)
    else:
        kq1, _ = R.quet_anh(a.index, a.matrix, de, a.k)

    # Dựng đáp án ĐÚNG NHƯ LÚC NỘP, rồi tra ngược về row_id để chấm
    ra = []
    for ten, c in ten_theo_cau.items():
        if c.loai == "TRAKE":
            dong_nop = R.dung_trake(kq1[ten], master, a.k)
        else:
            # ⚠️ `dap_an` o day quyet dinh con so QA co nghia gi. Truyen dap an
            # DUNG cua tap dev la cham nhu the VLM luon tra loi dung — TRAN
            # TREN, khong phai phep do. Phai chon co y thuc, khong mac dinh am.
            if c.loai == "QA" and a.dap_an == "vlm":
                from tra_loi import tra_loi_qa
                anh = [master.kf_path.iloc[x.row_id] for x in kq1[ten][:3]]
                da = tra_loi_qa(c.cau_hoi, [x for x in anh if isinstance(x, str)])
            elif c.loai == "QA" and a.dap_an == "trong":
                da = ""
            else:
                da = c.dap_an
            dong_nop = tu_ung_vien(kq1[ten], c.loai.lower(),
                                   dap_an=da, gioi_han=a.k)
        ra.append((c, dong_nop))

    nhan = {"hoan-hao": "TRẦN TRÊN (giả định VLM luôn đúng)",
            "trong": "SÀN DƯỚI (chưa có VLM)",
            "vlm": "VLM thật"}[a.dap_an]
    print(f"Đáp án Q&A: {nhan}\n")
    print(f"{'mức':>7}  {'KIS':>8}  {'QA':>8}  {'TRAKE':>8}  {'TỔNG':>8}")
    print("-" * 50)
    for ds in a.dung_sai:
        theo = defaultdict(list)
        for c, dong_nop in ra:
            if c.loai == "TRAKE":
                dung = [no_cua_so(b, master, ds) for b in c.row_id_dung]
                # mỗi dòng nộp: (video, f1..fN) -> row_id của từng vị trí
                dong_rid = []
                for x in dong_nop:
                    if not isinstance(x, AnswerTRAKE):
                        continue
                    dong_rid.append([
                        next((r for r in tra.get((x.video_id, f), [])
                              if r in dung[i]), -1)
                        for i, f in enumerate(x.frame_idxs[:len(dung)])])
                d = diem_trake_bai_nop(dong_rid, dung, a.k)
            else:
                dung = no_cua_so(c.row_id_dung, master, ds)
                # đổi đáp án nộp về Candidate để dùng lại `_hang`
                from schema import Candidate
                uv = []
                for x in dong_nop:
                    rid = next((r for r in tra.get((x.video_id, x.frame_idx), [])
                                if r in dung), None)
                    uv.append(Candidate(
                        row_id=rid if rid is not None else -1,
                        video_id=x.video_id, frame_idx=x.frame_idx, score=0.0,
                        meta={"answer": getattr(x, "answer", None)}))
                d = diem_cau(_hang(uv, dung, a.k, _dung_dap_an(c)))
            theo[c.loai].append(d)
            theo["TỔNG"].append(d)
        print(f"  ±{ds:<4g}s"
              + "".join(f"{(sum(theo[k]) / len(theo[k]) if theo[k] else 0):>10.4f}"
                        for k in ("KIS", "QA", "TRAKE", "TỔNG")))

    print("\n⚠️  Đây là điểm của THỨ ĐƯỢC NỘP, đã đi qua cả tầng lắp ráp và vòng\n"
          "    quay (video_id, frame_idx) -> row_id. Lệch so với A17 là do tầng\n"
          "    đó, không phải do kênh.")
    n_trake = sum(1 for c, _ in ra if c.loai == "TRAKE")
    if n_trake < 10:
        print(f"⚠️  Chỉ {n_trake} câu TRAKE — dưới mọi ngưỡng nhiễu. Cột TRAKE là\n"
              "    phép kiểm NỐI DÂY, không phải phép đo.")


if __name__ == "__main__":
    main()
