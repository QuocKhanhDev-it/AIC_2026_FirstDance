"""
67_gop_cache_truy_van.py — Gộp một file vector truy vấn mới vào cache đang dùng.

    python scripts/67_gop_cache_truy_van.py index/truy_van_de_thi_thu.npz
    python scripts/67_gop_cache_truy_van.py index/truy_van_de_thi_thu.npz --ghi

VÌ SAO GỘP Ở ĐÂY CHỨ KHÔNG GỘP TRÊN KAGGLE

`25_ma_hoa_truy_van.py --gop` gộp được, nhưng muốn thế thì phải đưa cache cũ
LÊN Kaggle rồi tải bản gộp VỀ. Hai lần truyền file cho 63 chuỗi mới, và mỗi
lần đều có cơ hội ghi đè nhầm — mất cache cũ là **mọi script đo đều tắc**, vì
máy 7,7 GB không nạp nổi model để sinh lại.

Ở đây: Kaggle chỉ sinh file MỚI (không cần biết cache cũ tồn tại), máy này gộp.
Cache cũ không bao giờ rời khỏi máy.

BỐN THỨ PHẢI ĐÚNG TRƯỚC KHI GHI

  * cùng model và cùng số chiều — trộn hai không gian vector là hỏng câm
  * số chuỗi phải TĂNG, không được giảm
  * chuỗi trùng thì giữ bản CŨ (vector đã dùng để đo, đừng đổi giữa chừng)
  * sao lưu bản cũ trước khi ghi đè
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parent.parent


def _doc(f: Path):
    z = np.load(f, allow_pickle=False)
    return ([str(c) for c in z["cau"]], np.asarray(z["vec"]),
            json.loads(str(z["ghi_chu"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("moi", type=Path, help="file .npz mới (từ Kaggle)")
    ap.add_argument("--cache", default=GOC / "index" / "truy_van_gopt.npz",
                    type=Path)
    ap.add_argument("--ghi", action="store_true",
                    help="ghi thật (mặc định chỉ xem trước)")
    a = ap.parse_args()

    cau_cu, vec_cu, gc_cu = _doc(a.cache)
    cau_moi, vec_moi, gc_moi = _doc(a.moi)
    print(f"cache : {len(cau_cu):,} chuỗi, {vec_cu.shape[1]} chiều "
          f"| {gc_cu.get('model')}")
    print(f"mới   : {len(cau_moi):,} chuỗi, {vec_moi.shape[1]} chiều "
          f"| {gc_moi.get('model')}")

    if vec_cu.shape[1] != vec_moi.shape[1]:
        raise SystemExit(f"❌ {vec_cu.shape[1]} chiều so với {vec_moi.shape[1]} "
                         f"— hai không gian vector khác nhau, KHÔNG gộp được.")
    if gc_cu.get("model") != gc_moi.get("model"):
        raise SystemExit(
            f"❌ model khác nhau: {gc_cu.get('model')!r} so với "
            f"{gc_moi.get('model')!r}. Cùng số chiều không có nghĩa là cùng "
            f"không gian — gộp vào là hỏng câm, không có gì báo.")

    # Trùng thì GIỮ BẢN CŨ: vector cũ là thứ mọi phép đo đã chạy trên đó.
    co = {c: i for i, c in enumerate(cau_cu)}
    them = [i for i, c in enumerate(cau_moi) if c not in co]
    print(f"\nthêm  : {len(them):,} chuỗi mới "
          f"({len(cau_moi) - len(them):,} chuỗi đã có, giữ bản cũ)")
    if not them:
        print("Không có gì để thêm. Dừng.")
        return

    cau = cau_cu + [cau_moi[i] for i in them]
    vec = np.vstack([vec_cu, vec_moi[them]])
    print(f"sau   : {len(cau):,} chuỗi")
    assert len(cau) > len(cau_cu) and len(cau) == len(vec)

    for c in cau_moi[:2]:
        print(f"   vd: {c[:88]!r}")

    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi thật)")
        return

    luu = a.cache.with_suffix(".npz.truoc_khi_gop")
    shutil.copy(a.cache, luu)
    np.savez(a.cache, cau=np.array(cau), vec=vec.astype(np.float32),
             ghi_chu=json.dumps({**gc_cu, "so_cau": len(cau)},
                                ensure_ascii=False))
    print(f"\n✅ {a.cache}  ({len(cau):,} chuỗi)")
    print(f"   bản cũ: {luu.name}")


if __name__ == "__main__":
    main()
