"""
105_danh_dau_nhan_sai.py — Hạ nhãn những câu BTC đã chấm 0 mà ta soi ra "đáp án".

    python scripts/105_danh_dau_nhan_sai.py            # xem trước
    python scripts/105_danh_dau_nhan_sai.py --ghi

VÌ SAO — VÒNG LẶP KHÉP KÍN TRONG CHÍNH THƯỚC ĐO

`66_soat_de_thi_thu.py` tìm đáp án bằng cách cho người soi **top ứng viên của
chính hệ thống** rồi bấm chọn khung trông đúng. Docstring của nó đã cảnh báo:
*"ĐÁP ÁN TÌM THẤY LÀ NIỀM TIN, KHÔNG PHẢI SỰ THẬT"*, và cả 20 câu đều dừng ở
nhãn `do_chac: kha — CHUA doi chieu anh goc`.

Bảng điểm THẬT của BTC (Sơ tuyển 1, 14,5/25) nay cho biết sáu câu trong số đó
được **0 điểm** — trong khi hệ thống xếp "đáp án" ta soi ở hạng 1, 2, 6, 6.
Nghĩa là người soi đã chọn một khung *trông hợp lý* từ top-20, và nó **không
phải** khoảnh khắc BTC chấm.

    p1-12  BTC 0 điểm — nhãn ta ở hạng 1
    p1-13  BTC 0 điểm — nhãn ta ở hạng 2
    p1-18  BTC 0 điểm — nhãn ta ở hạng 2
    p1-11  BTC 0 điểm — nhãn ta ở hạng 6
    p1-23  BTC 0 điểm — nhãn ta ở hạng 6
    p1-16  BTC 0 điểm — nhãn ta ở hạng 131

⚠️ THIÊN VỊ CÓ CHIỀU. Nhãn hái từ top-20 của cấu hình HIỆN TẠI thì bênh đúng
cấu hình đó, nên mọi cấu hình MỚI bị đẩy xuống. Đo được: bỏ 20 câu nhiễm khỏi
phép đo VietOCR (A88) làm kết luận đi từ ❌ ĐẢO DẤU sang 🟡, hiệu ở ±2s tăng
gấp đôi (+0,0076 -> +0,0144).

Script này chỉ hạ nhãn `do_chac`, **KHÔNG xoá câu**: câu hỏi vẫn dùng được nếu
sau này ai soi lại từ ẢNH GỐC theo mô tả. Xoá là mất luôn công đã bỏ ra.
"""

import argparse
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent

# BTC chấm 0 nhưng ta soi ra "đáp án" -> nhãn gần như chắc chắn SAI.
BTC_KHONG_DIEM = {"p1-11", "p1-12", "p1-13", "p1-16", "p1-18", "p1-23"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_thi_thu.jsonl",
                    type=Path)
    ap.add_argument("--ghi", action="store_true")
    a = ap.parse_args()

    dong = [json.loads(l) for l in a.file.read_text("utf-8").splitlines()
            if l.strip()]
    n = 0
    for d in dong:
        goi = "p1-" + str(d["id"]).rsplit("-", 1)[1]
        if goi in BTC_KHONG_DIEM and "do_chac: sai" not in d.get("ghi_chu", ""):
            d["ghi_chu"] = (
                "do_chac: sai — BTC chấm 0 điểm ở Sơ tuyển 1 nhưng nhãn này "
                "được soi ra từ top-20 của chính hệ thống (66_soat_de_thi_thu). "
                "KHÔNG dùng để đo cho tới khi soi lại từ ẢNH GỐC. Xem A89.")
            n += 1
            print(f"  hạ nhãn {d['id']}")

    print(f"\n{n}/{len(dong)} câu bị hạ nhãn")
    con = len(dong) - n
    print(f"{con} câu còn nhãn `kha` — vẫn CHƯA đối chiếu ảnh gốc, "
          f"chỉ là chưa có bằng chứng phản bác")
    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi thật)")
        return
    a.file.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dong) + "\n",
        encoding="utf-8", newline="\n")
    print(f"\n✅ {a.file}")


if __name__ == "__main__":
    main()
