"""
119_kiem_truy_van.py — TIỀN KIỂM trước khi chạy bài nộp: cache có đủ mọi chuỗi không?

    python scripts/119_kiem_truy_van.py --de dev/SOTUYEN3-bo-de

    exit 0 = đủ, chạy được       exit 1 = thiếu, IN RA lệnh vá

VÌ SAO CÓ FILE NÀY — MỘT CÂU MẤT TRẮNG Ở SƠ TUYỂN 2

`p2-22-kis` được 0 điểm với thông báo:

    Kênh 1 (ảnh) BỊ BỎ: truy vấn này chưa có trong index/truy_van.npz.
    Kết quả dưới đây chỉ từ kênh văn bản và objects — yếu hơn hẳn.

`run.py` có báo, và báo đúng. Nhưng nó báo **giữa lúc đang chạy cả bộ đề**, lẫn
trong hàng chục dòng log khác, và lúc ấy thì đã muộn: người chạy hoặc không kịp
thấy, hoặc thấy mà không còn thời gian mã hoá lại.

**Mất 1/30 câu = 3,3% bài thi, và mất vì VẬN HÀNH chứ không vì mô hình.** Đây
là loại mất mát phòng được 100%, nên nó xứng đáng có một chốt riêng chạy TRƯỚC.

CHỐT NÀY KHÁC CẢNH BÁO CỦA `run.py` Ở BA ĐIỂM

1. Chạy **trước**, khi vẫn còn thời gian mã hoá lại.
2. Kiểm **mọi chuỗi**, không chỉ chuỗi đầu tiên gặp — nên biết được thiếu 1 câu
   hay thiếu 20 câu, hai tình huống cần phản ứng khác hẳn nhau.
3. **In sẵn lệnh vá**, dán chạy được ngay, không phải tự gõ lại câu tiếng Việt
   có dấu (một chỗ dễ gõ sai mà sai thì lại thiếu tiếp).

⚠️ DÙNG CHÍNH `25_ma_hoa_truy_van.thu_thap()` để sinh danh sách chuỗi cần có.
Viết lại logic tách mệnh đề ở đây là mở đường cho hai bản lệch nhau — mà lệch
ở đúng chỗ này thì chốt báo "đủ" trong khi thật ra thiếu, tức tệ hơn không có
chốt.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
from dense import KenhAnhCache                        # noqa: E402

_sp = importlib.util.spec_from_file_location(
    "ma_hoa_25", GOC / "scripts" / "25_ma_hoa_truy_van.py")
_m = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_m)
thu_thap = _m.thu_thap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--cache", default=None, type=Path,
                    help="mặc định index/truy_van_gopt.npz")
    ap.add_argument("--matrix", default="clip_gopt.npy")
    ap.add_argument("--de", type=Path, help="thư mục chứa query-*.txt")
    ap.add_argument("--tap", action="append", default=[], type=Path)
    ap.add_argument("--tap-dev", action="store_true")
    a = ap.parse_args()

    cache = a.cache or (a.index / "truy_van_gopt.npz")
    if not cache.exists():
        raise SystemExit(f"❌ KHÔNG CÓ CACHE: {cache}\n"
                         f"   Kênh 1 sẽ TẮT HOÀN TOÀN, không phải thiếu vài câu.")
    if not (a.de or a.tap or a.tap_dev):
        raise SystemExit("Chưa chọn nguồn: --de / --tap / --tap-dev")

    can = thu_thap(a.de, a.tap_dev, [], a.tap)
    k1 = KenhAnhCache(str(a.index), str(cache), matrix=a.matrix)
    thieu = k1.co_du(can)

    print(f"\ncache : {cache}")
    print(f"nguồn : {a.de or a.tap or 'tập dev'}")
    print(f"cần   : {len(can)} chuỗi (câu gốc + mệnh đề + sự kiện TRAKE)")
    print(f"thiếu : {len(thieu)}\n")

    if not thieu:
        print("✅ ĐỦ — kênh 1 sẽ chạy cho MỌI truy vấn. Chạy bài nộp được.")
        return

    # Gói nào bị ảnh hưởng — quan trọng hơn danh sách chuỗi, vì điểm tính theo
    # GÓI chứ không theo chuỗi.
    goi_hong = set()
    if a.de:
        for ten, nd in R.doc_de(a.de).items():
            cac = ([m for sk in R.tach_su_kien(nd)
                    for m in [sk] + R.tach_truy_van(sk)]
                   if R.loai_cua(ten) == "trake"
                   else [nd] + R.tach_truy_van(nd))
            if any(c in thieu for c in cac):
                goi_hong.add(ten)

    print(f"❌ THIẾU {len(thieu)} chuỗi"
          + (f", ảnh hưởng {len(goi_hong)} GÓI:\n   " + ", ".join(sorted(goi_hong))
             if goi_hong else "") + "\n")
    for c in thieu[:10]:
        print(f"   • {c[:100]}")
    if len(thieu) > 10:
        print(f"   … và {len(thieu) - 10} chuỗi nữa")

    print("\n" + "=" * 70)
    print("VÁ NGAY — dán nguyên dòng dưới (máy ≥16 GB hoặc Kaggle):")
    print("=" * 70)
    nguon = (f"--de {a.de.as_posix()}" if a.de else
             " ".join(f"--tap {t.as_posix()}" for t in a.tap) or "--tap-dev")
    print(f"\n  python scripts/25_ma_hoa_truy_van.py {nguon} \\\n"
          f"      --matrix {a.matrix} --ra {cache.as_posix()} --gop\n")
    print("Rồi chạy lại chính lệnh này để xác nhận trước khi nộp.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
