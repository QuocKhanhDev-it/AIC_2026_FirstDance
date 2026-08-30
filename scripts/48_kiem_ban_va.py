"""
48_kiem_ban_va.py — Xác nhận một bản vá encode ĐÚNG phần được giao.

    python scripts/48_kiem_ban_va.py index/clip_gopt_phan3.npy
    python scripts/48_kiem_ban_va.py index/*.npy --nhom L26

VÌ SAO CẦN, KHI ĐÃ CÓ SIDECAR VÀ `18_ghep_ma_tran.py`

Sidecar chỉ ghi **số** dòng. Hai người lỡ chạy trùng một phần thì cả hai file
đều có ~16.290 vector, sidecar giống nhau, `18_ghep_ma_tran.py` ghép êm — và
kết quả là **một phần bị làm hai lần, một phần không ai làm**. Ma trận cuối
vẫn hợp lệ hoàn toàn, chỉ thiếu vector ở những dòng không ai đụng tới, và
không có phép kiểm nào phía sau bắt được.

Chỉ so **danh sách `row_id`** với phép chia mới biết ai làm phần nào.

Đã cắn hụt thật (30/08): hai sidecar về tới nơi bị hỏng và ghi cùng một con số
`16299` cho hai phần khác nhau. Ma trận thì đúng — nhưng chỉ script này chứng
minh được điều đó.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent


def _nap_chia():
    """Nạp `47_chia_viec_encode.py` để dùng ĐÚNG cùng phép chia."""
    f = GOC / "scripts" / "47_chia_viec_encode.py"
    s = importlib.util.spec_from_file_location("chia_viec", f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def be_co_vector(f: Path) -> set:
    """`row_id` có vector khác 0.

    Chỉ đọc 8 chiều đầu: đủ để biết dòng đã encode hay chưa (vector chuẩn hoá
    L2 thì không thể có 8 chiều đầu bằng 0 hết), và tránh kéo 545 MB qua mmap
    chỉ để đếm.
    """
    a = np.load(f, mmap_mode="r")
    if a.ndim != 2:
        raise SystemExit(f"❌ {f.name}: không phải ma trận 2 chiều {a.shape}")
    co = np.abs(np.asarray(a[:, :8], dtype=np.float32)).sum(1) > 0
    return set(np.flatnonzero(co).tolist()), a.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--nguoi", type=int, default=6)
    ap.add_argument("--nhom", default=None, metavar="L26",
                    help="phép chia riêng trong một nhóm L (xem 47_ --nhom)")
    a = ap.parse_args()

    cv = _nap_chia()
    master = pd.read_parquet(a.index / "master.parquet")
    m = master
    if a.nhom:
        nh = {x.strip().upper() for x in a.nhom.split(",")}
        m = master[master.video_id.str[:3].isin(nh)]
    phan, _ = cv.chia(m, a.nguoi, chi_co_anh=not a.nhom)
    can = [set(m[m.video_id.isin(p) & (m.kf_path.notna() if not a.nhom else True)].row_id)
           for p in phan]

    print(f"phép chia: {a.nguoi} phần"
          f"{f', riêng nhóm {a.nhom.upper()}' if a.nhom else ''}\n")
    print(f"{'file':<32}{'dòng':>9}  kết luận")
    print("-" * 74)

    thay: dict[int, list[str]] = {}
    for f in a.files:
        if not f.exists():
            print(f"{f.name:<32}{'':>9}  ❌ không có file")
            continue
        rid, shape = be_co_vector(f)
        kq = []
        for i, c in enumerate(can, 1):
            if rid == c:
                kq.append(f"✅ phần {i} — khớp tuyệt đối")
                thay.setdefault(i, []).append(f.name)
            elif rid & c:
                kq.append(f"phần {i}: {len(rid & c):,}/{len(c):,}")
        print(f"{f.name:<32}{len(rid):>9,}  "
              f"{' | '.join(kq) or '❌ KHÔNG khớp phần nào'}")

    print("-" * 74)
    loi = 0
    for i in sorted(thay):
        if len(thay[i]) > 1:
            loi += 1
            print(f"❌ PHẦN {i} BỊ LÀM {len(thay[i])} LẦN: {', '.join(thay[i])}")
    if loi:
        print("\n   Hai người cùng chạy một số. Phần không ai làm sẽ để trống mà\n"
              "   ma trận ghép ra vẫn hợp lệ — kiểm lại ai nhận số nào.")
        raise SystemExit(1)

    if thay:
        thieu = [i for i in range(1, a.nguoi + 1) if i not in thay]
        print(f"đã có: phần {sorted(thay)}")
        print(f"còn thiếu: phần {thieu}" if thieu else "ĐỦ CẢ %d PHẦN." % a.nguoi)


if __name__ == "__main__":
    main()
