r"""
46_de2_thanh_tap_do.py — Đề sơ tuyển đợt 2 + đáp án nhóm soát -> câu đo.

    python scripts/46_de2_thanh_tap_do.py            # xem trước
    python scripts/46_de2_thanh_tap_do.py --ghi      # ghi tap_dev_DE2.jsonl
    python scripts/46_de2_thanh_tap_do.py --ghi --loc-de-that   # + tap_de_that.jsonl

Nối tiếp `34_de_thanh_tap_dev.py` (đợt 1). Giữ nguyên ba kỷ luật ở đó, thêm hai
cái nữa mà đợt 2 mới sinh ra.

ĐÁP ÁN Ở ĐÂY LÀ NIỀM TIN, KHÔNG PHẢI SỰ THẬT
============================================

Bài nộp đợt 2 được **11,6 điểm** — phần lớn đúng, nhưng BTC **không nói câu nào
sai**. Một đáp án sai nằm trong thước đo thì mọi phép đo sau đó lệch mà không có
gì báo. Nên mỗi câu mang một nhãn `do_chac`, và phép đo nghiêm túc chỉ nên chạy
trên nhãn `xong`:

    xong        nhóm mở ảnh soát tay, hoặc tôi tự tra được bằng OCR/ảnh gốc
    kha         khớp nhiều chi tiết nhưng chưa mở hết
    cho-check   nhóm tự đánh dấu CHECK
    doan        không ai soi, chỉ là ứng viên tốt nhất
    mau-thuan   hai nguồn chỉ hai video khác nhau -> KHÔNG dùng để đo

HAI ĐIỀU RIÊNG CỦA ĐỢT 2
========================

**1. Chỉ lấy KHỐI ĐẦU của mỗi gói.** File `vqa/dap_an_p2.json` có gói mang nhiều
khối video: khối đầu là đáp án, các khối sau là **bảo hiểm** xếp thêm cho đủ 100
dòng. Đưa cả vào làm "đáp án đúng" là tự nới rộng đích cho tới khi bắn kiểu gì
cũng trúng.

**2. `query-p2-22-kis` bị đánh `mau-thuan` và loại khỏi tập đo.** Nhóm chốt
`L23_V013` (video đua xe đạp); tôi đọc đề thấy hỏi mực khứa vuông góc và chỉ vào
`L26_V177` "MỰC XÀO XỐT TIÊU XANH". Chưa ai phân xử được, mà đưa vào thì một
trong hai phía sẽ làm lệch thước đo.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                    # noqa: E402

DE = GOC / "dev" / "SOTUYEN2-bo-de-thi"
DAP_AN = GOC / "vqa" / "dap_an_p2.json"
RA = GOC / "dev" / "tap_dev_thanh_vien" / "tap_dev_DE2.jsonl"

# Nhãn từ danh sách cuối của nhóm, cộng thêm những gì tôi tự tra được.
DO_CHAC = {
    "query-p2-1-kis": "xong", "query-p2-3-kis": "xong", "query-p2-4-kis": "xong",
    "query-p2-6-kis": "xong", "query-p2-9-qa": "xong", "query-p2-10-kis": "xong",
    "query-p2-12-qa": "xong", "query-p2-14-kis": "xong", "query-p2-15-kis": "xong",
    "query-p2-16-kis": "xong", "query-p2-18-kis": "xong", "query-p2-26-kis": "xong",
    # tôi tự tra được, ghi rõ bằng cách nào ở `ghi_chu`
    "query-p2-7-qa": "xong", "query-p2-8-trake": "xong", "query-p2-17-kis": "xong",
    "query-p2-19-qa": "xong", "query-p2-27-qa": "xong", "query-p2-29-qa": "xong",
    "query-p2-5-kis": "kha", "query-p2-23-qa": "kha", "query-p2-28-qa": "kha",
    "query-p2-2-kis": "cho-check", "query-p2-11-kis": "cho-check",
    "query-p2-20-kis": "cho-check", "query-p2-24-kis": "cho-check",
    "query-p2-13-kis": "doan", "query-p2-30-qa": "doan",
    "query-p2-21-trake": "doan", "query-p2-25-kis": "xong",
    "query-p2-22-kis": "mau-thuan",
}

CACH_TRA = {
    "query-p2-7-qa": "mở ảnh gốc kf211: xe trắng mang số 1204, biển đỏ 6 ký tự chữ Hán",
    "query-p2-8-trake": "mở ảnh gốc kf159: ba trái sầu riêng trên cành",
    "query-p2-17-kis": "tra OCR 'SAC CO' trong phần chữ màn hình -> L30_V026, sheet thấy chữ nổi 3D 'SẮC CỔ VIỄN XƯA'",
    "query-p2-19-qa": "OCR kf30 đọc 'DC:552LyThungKiet,P.7Q.anbinh'; nhóm ghi kf163-171 nhưng video chỉ có 99 keyframe",
    "query-p2-27-qa": "đọc số trên cọc ở ảnh gốc kf1-4: thấy 1,3,4,5,6,7,8 -> thiếu 2",
    "query-p2-29-qa": "OCR bảng nguyên liệu: 'Thit be 200g'",
    "query-p2-28-qa": "OCR bảng nguyên liệu chỉ có 'Thit ca loc 300g', không có tôm",
    "query-p2-23-qa": "OCR ra L25_V012 câu 11 THPTQG 2022; đọc biểu đồ: loài (II) cực đại 6,5 phẳng từ 15 đến 20‰",
    "query-p2-5-kis": "sheet kf76-87: áo xanh đậm số 21 bám sát áo đen phối cam",
}

BO = {"query-p2-22-kis": "hai nguồn chỉ hai video khác nhau, chưa phân xử được"}


def main():
    ap = argparse.ArgumentParser(description="de dot 2 -> cau do")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--dap-an", default=DAP_AN, type=Path)
    ap.add_argument("--ra", default=RA, type=Path)
    ap.add_argument("--loc-de-that", action="store_true",
                    help="ghi thêm dev/tap_de_that.jsonl gồm mọi câu DE1+DE2")
    ap.add_argument("--ghi", action="store_true")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    fi = {(v, int(k)): int(r) for v, k, r
          in zip(master.video_id.values, master.kf_n.values, master.row_id.values)}
    tay = json.loads(a.dap_an.read_text("utf-8"))
    de = R.doc_de(DE)

    cau, bo_qua = [], []
    for ten in sorted(de):
        if ten in BO:
            bo_qua.append((ten, BO[ten]))
            continue
        m = tay.get(ten)
        if not m:
            bo_qua.append((ten, "chưa có đáp án"))
            continue

        # CHỈ khối đầu — các khối sau là bảo hiểm, không phải đáp án
        kh = m["khoi"][0] if "khoi" in m else m
        vid, kfs = kh["video"], list(kh["kf"])
        rid = [fi[(vid, int(k))] for k in kfs if (vid, int(k)) in fi]
        if not rid:
            bo_qua.append((ten, "không tra được row_id"))
            continue

        d = m.get("dap_an", "")
        dap = d if isinstance(d, str) else (d[0] if d else "")
        loai = R.loai_cua(ten)

        if loai == "trake":
            n = len(R.tach_su_kien(de[ten]))
            if len(rid) != n:
                bo_qua.append((ten, f"{len(rid)} khung nhưng đề có {n} sự kiện"))
                continue
            rd = [[r] for r in rid]
        else:
            rd = rid

        chac = DO_CHAC.get(ten, "doan")
        so = int(ten.split("-")[2])
        ghi = (f"CÂU ĐỀ THẬT do BTC viết ({len(de[ten].split())} từ). "
               f"Đáp án từ bài nộp đợt 2 (11,6 điểm) — BTC KHÔNG nói câu nào "
               f"sai, nên đây là NIỀM TIN chứ không phải sự thật. "
               f"Độ chắc: {chac}.")
        if ten in CACH_TRA:
            ghi += f" Cách tra: {CACH_TRA[ten]}."

        cau.append({
            "id": f"{loai}-DE2-{so:02d}",
            "loai": loai.upper(),
            "cau_hoi": de[ten],
            "row_id_dung": rd,
            "dap_an": dap,
            "nguon": f"đề sơ tuyển đợt 2, {vid} kf {kfs[0]}-{kfs[-1]}",
            "ghi_chu": ghi,
        })

    tu = [len(c["cau_hoi"].split()) for c in cau]
    print(f"{len(cau)} câu | trung vị {sorted(tu)[len(tu)//2]} từ/câu")
    print("  loại:", dict(collections.Counter(c["loai"] for c in cau)))
    print("  độ chắc:", dict(collections.Counter(
        DO_CHAC.get(f"query-p2-{int(c['id'].split('-')[2])}-{c['loai'].lower()}", "doan")
        for c in cau)))
    if bo_qua:
        print("\nBỎ:")
        for t, l in bo_qua:
            print(f"  {t}: {l}")

    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi thật)")
        return

    a.ra.parent.mkdir(parents=True, exist_ok=True)
    a.ra.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cau)
                    + "\n", "utf-8")
    print(f"\n✅ {a.ra}")
    print("   Gộp: python src/tap_dev.py --gop dev/tap_dev_thanh_vien")

    if a.loc_de_that:
        tap = GOC / "dev" / "tap_dev.jsonl"
        cu = [json.loads(l) for l in tap.read_text("utf-8").splitlines() if l.strip()]
        de1 = [c for c in cu if "-DE1-" in c["id"]]
        ra2 = GOC / "dev" / "tap_de_that.jsonl"
        ra2.write_text("\n".join(json.dumps(c, ensure_ascii=False)
                                 for c in de1 + cau) + "\n", "utf-8")
        print(f"✅ {ra2}  ({len(de1)} câu DE1 + {len(cau)} câu DE2 "
              f"= {len(de1) + len(cau)})")


if __name__ == "__main__":
    main()
