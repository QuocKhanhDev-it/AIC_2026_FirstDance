r"""
45_dong_goi_anh_kaggle.py — Đóng gói ảnh keyframe thành Private Dataset Kaggle.

    python scripts/45_dong_goi_anh_kaggle.py                  # xem trước
    python scripts/45_dong_goi_anh_kaggle.py --nhom L23 --ghi  # nén 1 nhóm
    python scripts/45_dong_goi_anh_kaggle.py --tat-ca --ghi    # nén hết

MỘT DATASET MỖI NHÓM L, KHÔNG PHẢI MỘT DATASET CHO CẢ KHO
=========================================================

Cả kho ≈ 34,5 GB / 177.321 file. Gộp làm một thì:

* upload đứt giữa chừng là mất cả, phải làm lại từ đầu;
* notebook nào cũng phải mount nguyên khối dù chỉ cần một nhóm;
* vượt hạn mức file của một dataset là hỏng lúc đang xử lý phía Kaggle.

Chia mười dataset thì mỗi cái 0,5–6 GB, hỏng cái nào làm lại cái đó, và
notebook mount đúng những nhóm nó cần. Kaggle cho nhiều nguồn dữ liệu mỗi
notebook nên mount cả mười cùng lúc vẫn được.

VÌ SAO NÉN THÀNH MỘT FILE TRƯỚC KHI UPLOAD
==========================================

Upload 79.590 file lẻ (L26) qua API là hàng chục nghìn request; nén thành một
`.zip` là một request. Kaggle tự giải nén phía nó sau khi nhận, nên trong
notebook vẫn thấy cây thư mục bình thường.

⚠️ JPEG đã nén sẵn — dùng `ZIP_STORED` (không nén lại). Nén lại tốn hàng chục
phút CPU để giảm được vài phần trăm.

RIÊNG TƯ — KHÔNG ĐƯỢC QUÊN
==========================

Đây là dữ liệu thi của BTC. `dataset-metadata.json` script này sinh ra đã đặt
sẵn để dataset là **private**. Sau khi tạo xong vẫn phải **mở trang dataset kiểm
lại nhãn Private** — đặt nhầm public là phát tán dữ liệu của BTC.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent


def goc_anh(master: pd.DataFrame, nhom: str) -> Path | None:
    """Thư mục `Keyframes_<nhóm>` suy ra từ `kf_path` thật trong bảng cái."""
    s = master[(master.video_id.str.startswith(nhom)) & (master.kf_path.notna())]
    if s.empty:
        return None
    p = Path(str(s.kf_path.iloc[0]))
    for cha in p.parents:
        if cha.name.lower().startswith("keyframes_"):
            return cha
    return None


def main():
    ap = argparse.ArgumentParser(description="dong goi anh keyframe len Kaggle")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--ra", default=GOC / "kaggle_upload", type=Path)
    ap.add_argument("--nhom", default=None, help="một nhóm, vd L23")
    ap.add_argument("--tat-ca", action="store_true")
    ap.add_argument("--user", default="<kaggle-username>",
                    help="username Kaggle, đi vào dataset-metadata.json")
    ap.add_argument("--ghi", action="store_true")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    co = master[master.kf_path.notna()]
    nhoms = sorted(co.video_id.str[:3].unique())
    if a.nhom:
        nhoms = [a.nhom] if a.nhom in nhoms else []
        if not nhoms:
            raise SystemExit(f"{a.nhom} chưa có ảnh ở máy này")
    elif not a.tat_ca:
        pass

    thieu = sorted(set(master.video_id.str[:3].unique()) - set(co.video_id.str[:3].unique()))
    print(f"{len(co):,}/{len(master):,} khung có ảnh"
          f"{'  |  CHƯA CÓ: ' + ', '.join(thieu) if thieu else ''}\n")

    tong_mb = 0
    for n in nhoms:
        g = goc_anh(master, n)
        if g is None or not g.exists():
            print(f"  {n}: không tìm thấy thư mục ảnh")
            continue
        anh = sorted(g.rglob("*.jpg"))
        mb = sum(p.stat().st_size for p in anh) / 1024 ** 2
        tong_mb += mb
        print(f"  {n}: {len(anh):>6,} ảnh  {mb:>8,.0f} MB   {g}")

        if not a.ghi:
            continue

        thu_muc = a.ra / f"aic2026-keyframes-{n.lower()}"
        thu_muc.mkdir(parents=True, exist_ok=True)
        (thu_muc / "dataset-metadata.json").write_text(json.dumps({
            "title": f"AIC2026 keyframes {n}",
            "id": f"{a.user}/aic2026-keyframes-{n.lower()}",
            "licenses": [{"name": "unknown"}],
            "isPrivate": True,
        }, indent=1), "utf-8")

        z = thu_muc / f"Keyframes_{n}.zip"
        if z.exists():
            print(f"      (đã có {z.name}, bỏ qua)")
            continue
        print(f"      nén -> {z.name} …", end="", flush=True)
        # ZIP_STORED: JPEG nén sẵn rồi, nén lại chỉ tốn CPU
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED, allowZip64=True) as f:
            for p in anh:
                f.write(p, p.relative_to(g.parent).as_posix())
        print(f" xong {z.stat().st_size / 1024 ** 2:,.0f} MB")

    print(f"\nTổng: {tong_mb:,.0f} MB")

    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để nén thật)")
        return

    print(f"""
Tiếp theo, ở máy này:

  pip install kaggle
  # tải kaggle.json ở kaggle.com -> Settings -> API -> Create New Token
  # đặt vào  C:\\Users\\<ten>\\.kaggle\\kaggle.json

  # tạo từng dataset (chạy lần đầu cho mỗi nhóm)
  kaggle datasets create -p "{a.ra}\\aic2026-keyframes-l23" --dir-mode zip

  # nếu cần đẩy lại nhóm đó về sau
  kaggle datasets version -p "{a.ra}\\aic2026-keyframes-l23" -m "cap nhat" --dir-mode zip

Sau mỗi dataset: MỞ TRANG DATASET, KIỂM NHÃN **Private**.
""")


if __name__ == "__main__":
    main()
