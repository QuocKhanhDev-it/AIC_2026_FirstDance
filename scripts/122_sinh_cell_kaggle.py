"""
122_sinh_cell_kaggle.py — Sinh sẵn CELL KAGGLE từ thư mục đề. Không phải dán tay câu nào.

    python scripts/122_sinh_cell_kaggle.py --de dev/SOTUYEN3-bo-de-thi

Ghi ra một file `.py` chỉ để **copy toàn bộ, dán vào một cell Kaggle**.

VÌ SAO KHÔNG DÁN TAY

`notebooks/kaggle_ma_hoa_dot3.md` bảo dán từng gói vào một dict. Với 36 gói
tiếng Việt có dấu, mỗi lần dán tay là một lần có thể lệch một ký tự — mà lệch
một ký tự là **một chuỗi khác**, là trượt cache, là đúng lỗi đã làm mất trắng
`p2-22` ở Sơ tuyển 2 (A103).

Script này đọc thẳng file `.txt` và nhúng nội dung dưới dạng **JSON**, nên nội
dung đi qua nguyên vẹn: không phải lo dấu nháy, xuống dòng, hay ký tự đặc biệt
trong câu hỏi.

⚠️ TÊN GÓI ĐƯỢC GIỮ NGUYÊN. `run.py` đọc **loại câu từ tên file**
(`...-kis` / `-qa` / `-trake`), và tên gói cũng là tên file nộp cho BTC. Đổi
tên là hỏng cả hai chỗ, nên script không đụng vào tên.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402

KHO = "https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git"

MAU = '''# ==================================================================
# CELL KAGGLE — mã hoá {n} gói của {ten}
# Sinh tự động bởi scripts/122_sinh_cell_kaggle.py. ĐỪNG sửa tay.
# Chạy CELL 1 (cài đặt) trước cell này — xem notebooks/kaggle_ma_hoa_dot3.md
# ==================================================================
import json, pathlib, subprocess, sys

DE = json.loads(r"""{de_json}""")

d = pathlib.Path("/kaggle/working/aic/dev/{ten}")
d.mkdir(parents=True, exist_ok=True)
for ten, nd in DE.items():
    (d / f"{{ten}}.txt").write_text(nd, encoding="utf-8")
print(f"✅ ghi {{len(DE)}} gói vào {{d}}")

# --------------------------------------------------- mã hoá + TỰ KIỂM
def chay(*lenh):
    r = subprocess.run([sys.executable, *lenh], cwd="/kaggle/working/aic",
                       capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-2000:], file=sys.stderr)
    return r.returncode

ma = chay("scripts/25_ma_hoa_truy_van.py",
          "--de", "dev/{ten}",
          "--matrix", "clip_gopt.npy",
          "--ra", "index/truy_van_dot3.npz")
assert ma == 0, "mã hoá THẤT BẠI — đọc log ở trên"

kt = chay("scripts/119_kiem_truy_van.py",
          "--de", "dev/{ten}",
          "--cache", "index/truy_van_dot3.npz")
assert kt == 0, "TIỀN KIỂM BÁO THIẾU — đừng tải về, đọc log ở trên"

# Đóng gói bằng Python thuần, KHÔNG gọi lệnh `zip` của hệ thống — một phụ
# thuộc nữa là một chỗ nữa có thể hỏng lúc đang tính từng phút.
import zipfile
goc = pathlib.Path("/kaggle/working/aic")
with zipfile.ZipFile("/kaggle/working/dot3.zip", "w",
                     zipfile.ZIP_DEFLATED) as z:
    z.write(goc / "index/truy_van_dot3.npz", "index/truy_van_dot3.npz")
    for f in sorted((goc / "dev/{ten}").glob("query-*.txt")):
        z.write(f, "dev/{ten}/" + f.name)
kb = pathlib.Path("/kaggle/working/dot3.zip").stat().st_size / 1024
print(f"\\n📦 /kaggle/working/dot3.zip ({{kb:.0f}} KB) — bấm Output ở cột phải để tải về")
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--de", type=Path, required=True)
    ap.add_argument("--ra", type=Path, default=None,
                    help="mặc định <thư mục đề>/_cell_kaggle.py")
    a = ap.parse_args()

    fs = sorted(a.de.glob("query-*.txt"))
    if not fs:
        raise SystemExit(f"❌ không có file query-*.txt nào trong {a.de}")

    de, rong = {}, []
    for f in fs:
        nd = f.read_text("utf-8").strip()
        if not nd:
            rong.append(f.name)
        de[f.stem] = nd
    if rong:
        raise SystemExit(f"❌ {len(rong)} file RỖNG: {rong}")

    # ⚠️ `"""` trong câu hỏi sẽ phá chuỗi raw của cell. Chưa gặp bao giờ, nhưng
    # nếu gặp thì phải biết NGAY chứ không phải lúc dán vào Kaggle.
    js = json.dumps(de, ensure_ascii=False, indent=1)
    if '"""' in js or "\\" in js.replace('\\"', "").replace("\\n", ""):
        raise SystemExit("❌ nội dung đề chứa ký tự phá chuỗi raw — sửa tay")

    ra = a.ra or (a.de / "_cell_kaggle.py")
    ra.write_text(MAU.format(n=len(de), ten=a.de.name, de_json=js),
                  encoding="utf-8", newline="\n")

    loai = Counter(k.rsplit("-", 1)[1] for k in de)
    can = []
    for ten, nd in de.items():
        if R.loai_cua(ten) == "trake":
            for sk in R.tach_su_kien(nd):
                can += [sk] + R.tach_truy_van(sk)
        else:
            can += [nd] + R.tach_truy_van(nd)
    print(f"\n✅ {ra}")
    print(f"   {len(de)} gói  {dict(loai)}")
    print(f"   {len(set(can))} chuỗi sẽ phải mã hoá (câu gốc + mệnh đề "
          f"+ sự kiện TRAKE)")
    print(f"   {ra.stat().st_size / 1024:.0f} KB\n")
    print("LÀM GÌ TIẾP:")
    print("  1. Mở Kaggle, chạy CELL 1 trong notebooks/kaggle_ma_hoa_dot3.md")
    print(f"  2. Mở {ra}, COPY TOÀN BỘ, dán vào cell 2, chạy")
    print("  3. Tải dot3.zip về, rồi ở máy này:")
    print("       scripts/120_gop_cache.py index/truy_van_gopt.npz "
          "index/truy_van_dot3.npz")
    print(f"       scripts/119_kiem_truy_van.py --de {a.de.as_posix()}")


if __name__ == "__main__":
    main()
