"""
120_gop_cache.py — Gộp hai (hoặc nhiều) file cache vector truy vấn. KHÔNG cần model.

    python scripts/120_gop_cache.py index/truy_van_gopt.npz index/truy_van_dot3.npz

File ĐẦU TIÊN là đích: nó được gộp thêm rồi ghi đè. Các file sau chỉ đọc.

VÌ SAO CẦN, VÀ VÌ SAO KHÔNG DÙNG `--gop` CỦA `25_`

`25_ma_hoa_truy_van.py --gop` cũng gộp, nhưng nó gộp **trong lúc mã hoá** nên
phải nạp model — mà máy thi 7,7 GB không nạp nổi. Ngày thi thì đường đi là:
Kaggle mã hoá đề mới ra một `.npz` nhỏ, mang về, và **máy nhà chỉ cần nối hai
mảng numpy lại**. Không có việc gì cần tới model ở bước đó.

⚠️ CHỐT SỐ CHIỀU VÀ TÊN MODEL. Gộp vector 1536 chiều với vector 1152 chiều thì
`np.vstack` ném lỗi ngay — đó là may. Nguy hiểm hơn là gộp hai file **cùng số
chiều nhưng khác model**: nó chạy trót lọt và sinh ra một cache mà một nửa
vector nằm trong không gian khác. Cosine vẫn tính ra số, kết quả vẫn trông hợp
lệ, và không có gì báo. Nên `ghi_chu.model` phải khớp, và lệch là DỪNG.

⚠️ TRÙNG CHUỖI thì giữ bản của file ĐÍCH. Cùng một câu mã hoá hai lần bởi cùng
một model ra cùng một vector, nên chọn bản nào cũng như nhau — nhưng phải chọn
một cách TẤT ĐỊNH, để chạy lại hai lần cho ra hai file giống hệt.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def doc(f: Path) -> tuple[dict, dict]:
    z = np.load(f, allow_pickle=False)
    ghi_chu = json.loads(str(z["ghi_chu"]))
    return ({str(c): np.asarray(v, np.float32)
             for c, v in zip(z["cau"], z["vec"])}, ghi_chu)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("dich", type=Path, help="file cache ĐÍCH (bị ghi đè)")
    ap.add_argument("them", type=Path, nargs="+", help="file cache gộp vào")
    ap.add_argument("--thu", action="store_true",
                    help="chỉ báo cáo, không ghi")
    a = ap.parse_args()

    if not a.dich.exists():
        raise SystemExit(f"❌ không có {a.dich}")
    cu, gc_dich = doc(a.dich)
    print(f"đích  : {a.dich.name:<28} {len(cu):>5} chuỗi  "
          f"({gc_dich.get('model')}, {gc_dich.get('chieu')} chiều)")

    tong_moi = 0
    for f in a.them:
        if not f.exists():
            raise SystemExit(f"❌ không có {f}")
        moi, gc = doc(f)
        # ⚠️ Khác model mà cùng số chiều là hỏng IM LẶNG — xem docstring.
        for khoa in ("model", "chieu"):
            if gc.get(khoa) != gc_dich.get(khoa):
                raise SystemExit(
                    f"\n❌ LỆCH `{khoa}`: đích có {gc_dich.get(khoa)!r}, "
                    f"{f.name} có {gc.get(khoa)!r}.\n"
                    f"   DỪNG. Gộp hai không gian nhúng khác nhau vẫn chạy "
                    f"trót lọt và vẫn ra số,\n   nhưng mọi kết quả sau đó là "
                    f"vô nghĩa mà không có gì báo.")
        chua_co = {c: v for c, v in moi.items() if c not in cu}
        print(f"      + {f.name:<26} {len(moi):>5} chuỗi, "
              f"{len(chua_co):>5} chuỗi MỚI")
        cu.update(chua_co)
        tong_moi += len(chua_co)

    if a.thu:
        print(f"\n(--thu) sẽ thành {len(cu)} chuỗi, thêm {tong_moi}. Không ghi.")
        return

    cau = list(cu)
    np.savez_compressed(
        a.dich, cau=np.array(cau, dtype=object).astype(str),
        vec=np.vstack([cu[c] for c in cau]).astype(np.float32),
        ghi_chu=json.dumps(gc_dich, ensure_ascii=False))
    mb = a.dich.stat().st_size / 1024 ** 2
    print(f"\n✅ {a.dich} — {len(cau)} chuỗi (+{tong_moi}), {mb:.2f} MB")
    print("\nBƯỚC TIẾP THEO, đừng bỏ:")
    print("   python scripts/119_kiem_truy_van.py --de <thư mục đề>")


if __name__ == "__main__":
    main()
