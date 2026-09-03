"""
101_kiem_ocr_phan.py — Soát 12 phần VietOCR TRƯỚC khi ghép.

    python scripts/101_kiem_ocr_phan.py
    python scripts/101_kiem_ocr_phan.py --ghep      # ghép sau khi soát sạch

SÁU THỨ PHẢI ĐÚNG, cả sáu đều hỏng ÂM THẦM nếu sai

**1. Đúng phần được giao.** Ai quên đổi `PHAN` thì nộp về OCR của phần người
khác — file hợp lệ, tên đúng, nội dung sai. Chỉ lộ khi đối chiếu `video_id` với
`chia_ocr/phan_*.txt`.

**2. Không trùng nhau giữa các phần.** Trùng thì ghép xong một ảnh có hai bản
OCR, và kết quả phụ thuộc THỨ TỰ ghép.

**3. Không thiếu ảnh.** Phiên Kaggle chết giữa chừng thì `.jsonl` vẫn hợp lệ,
chỉ cụt đuôi. Đối chiếu với số ảnh THẬT của các video trong phần.

**4. Không rỗng hàng loạt.** Khung không có chữ thì OCR rỗng là đúng; nhưng cả
phần rỗng thì model hỏng hoặc ảnh không tra được.

**5. TỶ LỆ CÓ DẤU.** Đây là *lý do duy nhất* chạy VietOCR (A68: `ocr_text` cũ
chỉ 31% có dấu; A76 đo bản thử đưa khung đáp án từ 20% lên 82%). Phần nào tỷ lệ
có dấu thấp bất thường là phần chạy nhầm model hoặc nhầm cấu hình.

**6. Cùng tốc độ.** `giay_moi_anh` lệch mạnh giữa các phần = có phần chạy CPU,
tức có thể khác cấu hình. Không tự nó làm sai, nhưng là dấu hiệu đáng soi.

⚠️ Kaggle đổi đuôi file khi tải (`.jsonl` -> `.txt`/`.json`). Đọc theo NỘI DUNG
chứ không theo đuôi.

⚠️ GHÉP RA FILE RIÊNG `index/ocr_vietocr.parquet`, KHÔNG đè `ocr_asr.parquet`.
A76 đo được VietOCR **làm mất một con số** OCR cũ đọc được (`46`), nên cấu hình
thắng là **GỘP** hai nguồn chứ không phải thay.
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))


def co_dau(s) -> bool:
    return any(unicodedata.category(c) == "Mn"
               for c in unicodedata.normalize("NFD", str(s)))


def doc_phan(f: Path):
    """-> (danh sách dòng, meta). Đọc theo NỘI DUNG, bỏ qua đuôi file."""
    dong, meta = [], {}
    for l in f.read_text("utf-8").splitlines():
        if not l.strip():
            continue
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue                      # dòng cuối bị cắt khi phiên chết
        (meta.update(d) if d.get("_meta") else dong.append(d))
    return dong, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--thu-muc", default=GOC / "index" / "OCR_ASR", type=Path)
    ap.add_argument("--chia", default=GOC / "chia_ocr", type=Path)
    ap.add_argument("--ghep", action="store_true")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet",
                             columns=["row_id", "video_id", "kf_n"])
    # (video_id, kf_name) -> row_id. kf_name của bản chạy là "%03d.jpg" theo kf_n.
    tra = {(v, f"{int(n):03d}.jpg"): int(r) for r, v, n in
           zip(master.row_id.values, master.video_id.values, master.kf_n.values)}
    so_anh_video = master.video_id.value_counts().to_dict()

    cac = {}
    for f in sorted(a.thu_muc.iterdir()):
        if f.is_file() and "ocr_vietocr_phan" in f.name:
            cac[f.name.split("phan")[1].split(".")[0]] = f
    print(f"{len(cac)} phần trong {a.thu_muc}\n")

    print(f"{'phần':>5}{'dòng':>9}{'video':>7}{'đúng phần':>11}{'thiếu ảnh':>11}"
          f"{'rỗng':>8}{'CÓ DẤU':>9}{'s/ảnh':>8}")
    print("-" * 68)

    tat_ca, loi, da_thay = {}, [], {}
    for ten in sorted(cac, key=lambda x: (x[0], int(x[1:]))):
        dong, meta = doc_phan(cac[ten])
        giao = a.chia / f"phan_{ten}.txt"
        mong = ({l.strip() for l in giao.read_text("utf-8").splitlines()
                 if l.strip()} if giao.exists() else None)

        vids = {d["video_id"] for d in dong}
        dung_phan = mong is None or vids <= mong
        can = sum(so_anh_video.get(v, 0) for v in (mong or vids))
        rong = sum(1 for d in dong if not str(d.get("text", "")).strip())
        dau = sum(1 for d in dong if co_dau(d.get("text", "")))

        for d in dong:
            k = (d["video_id"], d["kf_name"])
            if k in da_thay and da_thay[k] != ten:
                loi.append(f"{ten}: {k} đã có ở phần {da_thay[k]}")
            da_thay[k] = ten

        if not dung_phan:
            loi.append(f"{ten}: có {len(vids - mong)} video KHÔNG thuộc phần "
                       f"này, vd {sorted(vids - mong)[:3]}")
        thieu = can - len(dong)
        if thieu > can * 0.02:
            loi.append(f"{ten}: thiếu {thieu:,}/{can:,} ảnh (>2%)")

        print(f"{ten:>5}{len(dong):>9,}{len(vids):>7}"
              f"{'✅' if dung_phan else '❌':>11}{thieu:>11,}"
              f"{rong / max(len(dong), 1) * 100:>7.0f}%"
              f"{dau / max(len(dong), 1) * 100:>8.0f}%"
              f"{meta.get('giay_moi_anh', float('nan')):>8.2f}")
        tat_ca[ten] = dong

    tong = sum(len(x) for x in tat_ca.values())
    print(f"\n{'TỔNG':>5}{tong:>9,} dòng / {len(master):,} keyframe "
          f"({tong / len(master) * 100:.1f}%)")

    if loi:
        print("\n❌ CÓ VẤN ĐỀ:")
        for x in loi[:20]:
            print(f"   {x}")
        raise SystemExit(1)
    print("\n✅ Mọi phần đều sạch.")

    if not a.ghep:
        print("   (thêm --ghep để gộp vào index/ocr_vietocr.parquet)")
        return

    ra, thieu_tra = [], 0
    for dong in tat_ca.values():
        for d in dong:
            r = tra.get((d["video_id"], d["kf_name"]))
            if r is None:
                thieu_tra += 1
                continue
            ra.append((r, str(d.get("text", "")).strip()))
    b = (pd.DataFrame(ra, columns=["row_id", "text"])
         .drop_duplicates("row_id", keep="last").sort_values("row_id"))
    f = a.index / "ocr_vietocr.parquet"
    if f.exists():
        f.replace(f.with_suffix(".parquet.truoc_khi_ghep"))
    b.to_parquet(f, index=False)
    print(f"\n✅ {f}: {len(b):,} khung"
          f" ({b.text.str.strip().ne('').sum():,} có chữ)")
    if thieu_tra:
        print(f"   ⚠️ {thieu_tra:,} dòng không tra được row_id — bỏ qua")


if __name__ == "__main__":
    main()
