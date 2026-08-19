r"""
18_ghep_ma_tran.py — Ghép một ma trận "vá" (encode riêng một lô video mới
tải) vào ma trận chính, KHÔNG chạy lại từ đầu.

Vì sao cần: `08_encode.py` đánh dấu dòng thiếu ảnh là "xong" (vector 0) ngay
khi chạy — nên chạy lại đúng lệnh cũ trên cùng `--out` sau khi tải thêm ảnh
KHÔNG tự vá được các dòng đó (đã coi là xong). Cách đúng: encode riêng lô mới
bằng `--chi-video` ra file khác, rồi ghép bằng script này.

    python scripts/18_ghep_ma_tran.py \
        --chinh index/clip_siglip2.npy \
        --va index/clip_siglip2_L26d_va.npy

Nguyên tắc ghép: dòng nào trong ma trận VÁ có vector khác 0 (đã encode thật,
không phải "chưa tải ảnh") thì ghi đè dòng đó vào ma trận CHÍNH. Dòng nào vẫn
0 ở ma trận vá thì bỏ qua — giữ nguyên ma trận chính, không ghi đè bằng 0.

BA ĐIỀU KHÔNG BAO GIỜ ĐỘNG TỚI, giống `12_va_duong_dan.py`:
  * Shape/dtype hai ma trận phải khớp — sai là DỪNG, không đoán.
  * Số dòng ma trận chính không đổi.
  * Dòng ngoài danh sách vá giữ nguyên giá trị cũ tuyệt đối — script tự kiểm.

Mặc định chỉ XEM TRƯỚC, không ghi. Thêm `--ghi` để ghi thật. Sao lưu bản cũ
thành `<tên>.truoc_khi_ghep.npy` trước khi ghi đè.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chinh", required=True, type=Path, help="ma trận chính, sẽ bị ghi đè")
    ap.add_argument("--va", required=True, type=Path, help="ma trận vá (encode riêng lô mới)")
    ap.add_argument("--ghi", action="store_true", help="ghi thật (mặc định chỉ xem trước)")
    a = ap.parse_args()

    chinh = np.load(a.chinh)
    va = np.load(a.va)
    print(f"chính {a.chinh}  {chinh.shape} {chinh.dtype}")
    print(f"vá    {a.va}  {va.shape} {va.dtype}")

    if chinh.shape != va.shape:
        raise SystemExit(f"❌ SHAPE LỆCH: {chinh.shape} vs {va.shape} — DỪNG, không đoán.")
    if chinh.dtype != va.dtype:
        raise SystemExit(f"❌ DTYPE LỆCH: {chinh.dtype} vs {va.dtype} — DỪNG, không đoán.")

    # dòng "đã encode thật" trong ma trận vá = có ít nhất một phần tử khác 0
    co_vector_va = np.abs(va).sum(axis=1) > 0
    n_va = int(co_vector_va.sum())
    if n_va == 0:
        raise SystemExit("❌ Ma trận vá không có dòng nào khác 0 — không có gì để ghép.")

    co_vector_chinh_truoc = np.abs(chinh).sum(axis=1) > 0
    moi = int((co_vector_va & ~co_vector_chinh_truoc).sum())
    de_truoc_giong_nhau = int((co_vector_va & co_vector_chinh_truoc).sum())

    print(f"\nDòng có vector thật trong bản vá: {n_va:,}")
    print(f"  trong đó MỚI (chính đang là 0):        {moi:,}")
    print(f"  trong đó GHI ĐÈ (chính đã có sẵn):      {de_truoc_giong_nhau:,}")

    ghep = chinh.copy()
    ghep[co_vector_va] = va[co_vector_va]

    # kiểm toàn vẹn: mọi dòng NGOÀI danh sách vá phải giữ nguyên tuyệt đối
    khong_dung = ~co_vector_va
    if not np.array_equal(ghep[khong_dung], chinh[khong_dung]):
        raise SystemExit("❌ CÓ DÒNG NGOÀI Ý MUỐN BỊ ĐỔI — KHÔNG GHI. Báo lỗi này lại, đừng tự sửa tay.")

    tong_truoc = int(co_vector_chinh_truoc.sum())
    tong_sau = int((np.abs(ghep).sum(axis=1) > 0).sum())
    print(f"\nTổng dòng có vector thật: {tong_truoc:,} -> {tong_sau:,}  (+{tong_sau - tong_truoc:,})")

    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi thật)")
        return

    sao_luu = a.chinh.with_suffix(a.chinh.suffix + ".truoc_khi_ghep")
    shutil.copy2(a.chinh, sao_luu)
    np.save(a.chinh, ghep)
    print(f"\n✅ Đã ghi {a.chinh}\n   sao lưu bản cũ: {sao_luu}")

    # cập nhật .json đi kèm nếu có
    canh = a.chinh.with_suffix(".json")
    if canh.exists():
        meta = json.loads(canh.read_text(encoding="utf-8"))
        meta["co_vector"] = tong_sau
        meta["ghep_them"] = meta.get("ghep_them", []) + [str(a.va)]
        canh.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   cập nhật {canh}: co_vector = {tong_sau:,}")

    print("\nBƯỚC TIẾP THEO — bắt buộc:")
    print(f"    python scripts/08_encode.py --kiem-lech-hang {a.chinh}")


if __name__ == "__main__":
    main()
