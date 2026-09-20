"""
117_soat_tay_de_thi_thu.py — Kết quả SOÁT TAY 24/25 gói đề thử nghiệm, và đối chiếu ba nguồn.

    python scripts/117_soat_tay_de_thi_thu.py

NGUỒN DỮ LIỆU, VÀ NÓ LÀ LOẠI GÌ

Người trong nhóm mở UI + dataset, tự chấm từng gói: **1 = đáp án nằm trong
top-20, 0 = ngoài**, kèm ghi chú vì sao trượt. Đây là dữ liệu quý vì nó cho
**KIỂU trượt**, thứ mà không bảng điểm nào cho.

⚠️ **NHƯNG ĐÂY KHÔNG PHẢI NHÃN VÀNG, VÀ SCRIPT NÀY TỒN TẠI ĐỂ NÓI RÕ ĐIỀU ĐÓ.**
Nó đo *"top-20 có chứa thứ trông giống đáp án theo mắt ta không"* — đúng cùng
đại lượng mà nhãn tự soi của `66_soat_de_thi_thu.py` đo, và A89 đã chứng minh
đại lượng đó **không khớp** điểm thật của BTC. Đối chiếu bên dưới cho thấy nó
khớp BTC **2/7**.

Nên: dùng cột `ghi_chu` để biết hệ thống hỏng KIỂU gì. **Đừng** dùng cột `soi`
làm đáp án đúng, và đừng nạp nó vào tập dev.
"""

import argparse
import math
from collections import Counter

# (điểm soi tay, loại, kiểu trượt, ghi chú nguyên văn)
# `kiểu trượt`: "" = không trượt | "hang" = đúng video sai khung/đáp án
#               "tim" = không tìm ra video | "doc" = không đọc được đáp án
SOAT = {
    "p1-1":  (1, "kis",   "",     ""),
    "p1-2":  (1, "kis",   "",     ""),
    "p1-4":  (1, "trake", "",     "đúng video + khung lân cận, cần mở dataset để chuẩn hơn"),
    "p1-5":  (1, "kis",   "",     ""),
    "p1-6":  (0, "kis",   "hang", "đúng video, KHÔNG ra đúng keyframe"),
    "p1-7":  (0, "kis",   "hang", "đúng video, KHÔNG ra đúng keyframe"),
    "p1-8":  (1, "kis",   "",     "không chắc lắm, đã mở dataset xem lại"),
    "p1-9":  (1, "kis",   "",     ""),
    "p1-10": (1, "kis",   "",     ""),
    "p1-11": (1, "kis",   "",     ""),
    "p1-12": (1, "kis",   "",     ""),
    "p1-13": (1, "kis",   "",     ""),
    "p1-14": (1, "kis",   "",     ""),
    "p1-15": (1, "qa",    "",     ""),
    "p1-16": (0, "trake", "tim",  "không tìm thấy đoạn video"),
    "p1-17": (1, "kis",   "",     ""),
    "p1-18": (0, "trake", "tim",  "không tìm thấy đoạn video"),
    "p1-19": (0, "qa",    "doc",  "không có keyframe chứa câu trả lời"),
    "p1-20": (1, "kis",   "",     "biết video + khung, không thấy rõ đoạn đặt 2 ly"),
    "p1-21": (0, "kis",   "tim",  "không tìm thấy đoạn video"),
    "p1-22": (0, "qa",    "hang", "đúng video có chứa đáp án, không ra đáp án"),
    "p1-23": (1, "kis",   "",     ""),
    "p1-24": (1, "kis",   "",     ""),
    "p1-25": (1, "kis",   "",     ""),
}

# Soát tay Sơ tuyển 2 (A103) — cùng cách, 30 gói. Ghi riêng vì đây là ĐỀ KHÁC,
# gộp chung vào SOAT là trộn hai bộ đề khác nhau.
SOAT2 = {
    "p2-1":  (1,    "kis",   ""),
    "p2-2":  (1,    "kis",   ""),
    "p2-3":  (0,    "kis",   "tim"),
    "p2-4":  (1,    "kis",   ""),
    "p2-5":  (0,    "kis",   "?"),
    "p2-6":  (1,    "kis",   ""),
    "p2-7":  (1,    "qa",    ""),
    "p2-8":  (1,    "trake", "hang"),
    "p2-9":  (0,    "qa",    "hang"),      # hạng 31
    "p2-10": (0.75, "kis",   "hang"),
    "p2-11": (0,    "kis",   "tim"),
    "p2-12": (1,    "qa",    "hang"),      # hạng 32
    "p2-13": (1,    "kis",   ""),
    "p2-14": (1,    "kis",   ""),
    "p2-15": (0,    "kis",   "tim"),
    "p2-16": (1,    "kis",   "hang"),      # hạng 24
    "p2-17": (1,    "kis",   ""),
    "p2-18": (1,    "kis",   ""),
    "p2-19": (1,    "qa",    ""),
    "p2-20": (1,    "kis",   ""),
    "p2-21": (0,    "trake", "hang"),
    "p2-22": (0,    "kis",   "LOI"),       # truy vấn CHƯA MÃ HOÁ
    "p2-23": (1,    "qa",    ""),
    "p2-24": (0,    "kis",   "tim"),
    "p2-25": (1,    "kis",   ""),
    "p2-26": (1,    "kis",   ""),
    "p2-27": (1,    "qa",    "hang"),
    "p2-28": (1,    "qa",    ""),
    "p2-29": (0,    "qa",    "tim"),
    "p2-30": (0,    "qa",    "tim"),
}

# ⚠️ p2-9 (hạng 31) bị chấm 0 còn p2-12 (hạng 32) được chấm 1. Theo công thức
# BTC cả hai đều trong (21,50] -> R@50 -> ĐỀU được 0,4. Thang "1 hay 0" không
# mô tả được cách chấm thật; khi soi tay thì GHI HẠNG dùng được hơn nhiều.

# Điểm THẬT của BTC ở Sơ tuyển 1 cho những gói đã biết (A89). Đây MỚI là nhãn
# vàng duy nhất repo có.
BTC = {"p1-6": 1, "p1-11": 0, "p1-12": 0, "p1-13": 0,
       "p1-16": 0, "p1-18": 0, "p1-23": 0}

# R@20 đo được trên 52 câu nhãn sạch (A92), để so với tỷ lệ soi tay.
R20_A92 = {"kis": 0.68, "qa": 0.42, "trake": 0.67}

TEN_KIEU = {"hang": "đúng video, sai khung/đáp án  -> lỗi XẾP HẠNG",
            "tim":  "không tìm thấy video          -> lỗi TÌM KIẾM",
            "doc":  "không đọc được đáp án         -> lỗi ĐỌC"}


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()
    n = len(SOAT)
    tot = sum(v[0] for v in SOAT.values())

    print(f"\n{'=' * 68}\nSOÁT TAY — {n}/25 gói (thiếu p1-3)\n{'=' * 68}")
    print(f"\nĐáp án nằm trong top-20: {tot}/{n} = {tot / n:.3f}\n")
    print(f"  {'loại':<8}{'đạt':>8}{'tỷ lệ':>9}")
    print("  " + "-" * 25)
    for loai in ("kis", "qa", "trake"):
        ds = [v for v in SOAT.values() if v[1] == loai]
        d = sum(v[0] for v in ds)
        print(f"  {loai.upper():<8}{f'{d}/{len(ds)}':>8}{d / len(ds):>9.3f}")

    print(f"\n{'-' * 68}\nKIỂU TRƯỢT — thứ mà bảng điểm không cho biết\n"
          f"{'-' * 68}")
    kieu = Counter(v[2] for v in SOAT.values() if v[0] == 0)
    for k, ten in TEN_KIEU.items():
        goi = [g for g, v in SOAT.items() if v[2] == k]
        print(f"  {kieu.get(k, 0)}  {ten}")
        for g in goi:
            print(f"       {g:<8}{SOAT[g][3]}")
    hang = kieu.get("hang", 0)
    print(f"\n  -> {hang}/{n - tot} câu trượt ĐÃ CÓ ĐÚNG VIDEO trong tay.")
    print(f"     A100 đo trần 0,7885 với 0,5231 đang đạt; con số này là cùng")
    print(f"     một sự thật, nhìn bằng mắt người thay vì bằng thước đo.")

    print(f"\n{'-' * 68}\nĐỐI CHIẾU 1 — thước đo có đoán đúng tỷ lệ này không?\n"
          f"{'-' * 68}")
    sl = Counter(v[1] for v in SOAT.values())
    du = sum(R20_A92[k] * sl[k] for k in sl) / n
    se = math.sqrt((tot / n) * (1 - tot / n) / n)
    print(f"  R@20 DỰ ĐOÁN từ A92 (trọng số {dict(sl)}) : {du:.3f}")
    print(f"  R@20 QUAN SÁT khi soi tay                 : {tot / n:.3f}")
    print(f"  lệch {tot / n - du:+.3f}, ngưỡng nhiễu 2×SE = {2 * se:.3f}"
          f"  -> {'TRONG' if abs(tot / n - du) < 2 * se else 'NGOÀI'} nhiễu")
    print("\n  Đây là lần kiểm ĐỘC LẬP thứ hai cho thước đo (sau bảng điểm BTC\n"
          "  ở A89), và lần này nó khớp.")

    print(f"\n{'-' * 68}\nĐỐI CHIẾU 2 — soi tay có khớp ĐIỂM THẬT của BTC không?\n"
          f"{'-' * 68}")
    print(f"  {'gói':<8}{'BTC':>5}{'soi tay':>9}   khớp")
    print("  " + "-" * 32)
    k = 0
    for g, b in sorted(BTC.items()):
        s = SOAT[g][0]
        k += (b == s)
        print(f"  {g:<8}{b:>5}{s:>9}   {'✓' if b == s else '✗'}")
    print(f"\n  KHỚP {k}/{len(BTC)} = {k / len(BTC):.2f}")
    print("\n  ⚠️ Và lệch CÓ HỆ THỐNG, không ngẫu nhiên: 4 câu BTC cho 0 thì\n"
          "     soi tay cho 1; còn p1-6 thì BTC cho 1 mà soi tay cho 0.\n"
          "     p1-6 là ca sắc nhất — bài nộp CÓ một khung trong cửa sổ của\n"
          "     BTC, mà cả nhãn tự soi (hạng 152, A89) lẫn mắt người đều\n"
          "     không nhận ra. Nghĩa là thứ ta tin là đáp án KHÔNG phải đáp\n"
          "     án của BTC.")

    # ------------------------------------------------ Sơ tuyển 2 (A103)
    n2 = len(SOAT2)
    t2 = sum(v[0] for v in SOAT2.values())
    print(f"\n{'=' * 68}\nSƠ TUYỂN 2 — {n2} gói\n{'=' * 68}")
    print(f"\nĐạt: {t2}/{n2} = {t2 / n2:.3f}\n")
    for loai in ("kis", "qa", "trake"):
        ds = [v for v in SOAT2.values() if v[1] == loai]
        d = sum(v[0] for v in ds)
        print(f"  {loai.upper():<8}{f'{d}/{len(ds)}':>10}{d / len(ds):>9.3f}")

    print(f"\n{'-' * 68}\nKIỂU TRƯỢT — CỘNG hai đợt (54 gói)\n{'-' * 68}")
    ten2 = {**TEN_KIEU, "LOI": "truy vấn CHƯA MÃ HOÁ         -> lỗi VẬN HÀNH",
            "?": "không chắc về nội dung        -> không phân loại được"}
    k1 = Counter(v[2] for v in SOAT.values() if v[2])
    k2 = Counter(v[2] for v in SOAT2.values() if v[2])
    print(f"  {'kiểu':<46}{'đợt1':>6}{'đợt2':>6}{'cộng':>6}")
    print("  " + "-" * 64)
    for k, ten in ten2.items():
        a, b = k1.get(k, 0), k2.get(k, 0)
        if a or b:
            print(f"  {ten:<46}{a:>6}{b:>6}{a + b:>6}")
    print(f"\n  -> Trên 54 gói, lỗi XẾP HẠNG ({k1.get('hang', 0) + k2.get('hang', 0)})"
          f" nhiều hơn lỗi TÌM KIẾM ({k1.get('tim', 0) + k2.get('tim', 0)}).")
    print("     A100 đo trần 0,7885 với 0,5231 đang đạt và kết luận 'bài toán")
    print("     còn lại là xếp lại hạng'. Đây là xác nhận thứ ba, mẫu gấp đôi.")
    if k2.get("LOI"):
        print(f"\n  🔴 {k2['LOI']} gói mất vì VẬN HÀNH (truy vấn chưa mã hoá) —")
        print("     phòng được 100%. Chạy scripts/119_kiem_truy_van.py TRƯỚC")
        print("     mỗi lần nộp. Xem docs/09_ngay_thi.md.")


if __name__ == "__main__":
    main()
