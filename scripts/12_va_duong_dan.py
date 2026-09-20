r"""
12_va_duong_dan.py — Vá lại `kf_path` / `video_path` trong `master.parquet`.

Sửa lỗi A5.5, thứ đã cản trở ba lần:

  1. Máy Khánh tải `index/` từ Drive — mọi đường dẫn trỏ vào ổ của máy khác
  2. Tải thêm `Keyframes_L27` — bảng cái vẫn ghi "chưa tải" dù ảnh đã trên đĩa
  3. Encode SigLIP2 bỏ qua hết L27 vì `kf_path` rỗng

Nguyên nhân: `01_build_index.py` ghi đường dẫn TUYỆT ĐỐI tại thời điểm dựng.
Nhóm L tải sau, hoặc `index/` chép sang máy khác, là đường dẫn sai hết.

    python scripts/12_va_duong_dan.py                 # xem trước, KHÔNG ghi
    python scripts/12_va_duong_dan.py --ghi           # ghi thật
    python scripts/12_va_duong_dan.py --roots D:\AIC  # nơi khác

BA ĐIỀU KHÔNG BAO GIỜ ĐỘNG TỚI:

  * `row_id` — cả hệ thống khóa vào nó. `clip.npy`, `objects.parquet`,
    `trung_lap.parquet`, và mọi `row_id_dung` trong tập dev đều tra theo nó.
  * SỐ DÒNG và THỨ TỰ DÒNG — chỉ sửa TẠI CHỖ vài cột đường dẫn.
  * Mọi cột khác — `video_id`, `kf_n`, `frame_idx`, `pts_time`, `fps`,
    metadata giữ nguyên tuyệt đối.

Script tự kiểm ba điều đó sau khi vá và **từ chối ghi nếu sai một điều**.
Sao lưu bản cũ thành `master.parquet.truoc_khi_va` trước khi ghi đè.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
MAC_DINH_ROOTS = [r"C:\Code\aic_data", r"D:\Project\AIC_2026\AIC_data"]


def quet_keyframe(roots) -> dict:
    """`video_id` -> thư mục chứa ảnh keyframe của video đó.

    Quét đệ quy và nhận diện theo TÊN THƯ MỤC khớp `L\\d\\d_V\\d+`, không ép
    cấu trúc gói tải về vào khuôn nào — cùng triết lý `00_discover.py`.
    """
    import re
    mau = re.compile(r"^L\d{2}_V\d+$")
    ra = {}
    for g in roots:
        g = Path(g)
        if not g.exists():
            continue
        for d in g.rglob("*"):
            if d.is_dir() and mau.match(d.name) and d.name not in ra:
                if any(d.glob("*.jpg")):
                    ra[d.name] = d
    return ra


def quet_video(roots) -> dict:
    """`video_id` -> file `.mp4`."""
    ra = {}
    for g in roots:
        g = Path(g)
        if not g.exists():
            continue
        for f in g.rglob("*.mp4"):
            ra.setdefault(f.stem, f)
    return ra


def va(master: pd.DataFrame, thu_muc: dict, video: dict) -> tuple[pd.DataFrame, dict]:
    """Trả về (bảng đã vá, thống kê). KHÔNG sửa `master` gốc."""
    m = master.copy()
    kf_path, kf_name, vid_path = [], [], []
    dem = {"anh_moi": 0, "anh_giu": 0, "anh_khong_co": 0,
           "video_moi": 0, "video_giu": 0, "video_khong_co": 0}

    for r in m.itertuples():
        # --- ảnh keyframe ---
        d = thu_muc.get(r.video_id)
        ten = f"{int(r.kf_n):03d}.jpg"
        p = d / ten if d else None
        if p is not None and p.exists():
            kf_path.append(str(p))
            kf_name.append(ten)
            dem["anh_moi" if str(p) != str(r.kf_path) else "anh_giu"] += 1
        elif isinstance(r.kf_path, str) and Path(r.kf_path).exists():
            kf_path.append(r.kf_path)          # đường cũ vẫn đúng, giữ nguyên
            kf_name.append(r.kf_name if isinstance(r.kf_name, str) else ten)
            dem["anh_giu"] += 1
        else:
            kf_path.append(None)               # thật sự chưa có -> để trống
            kf_name.append(None)
            dem["anh_khong_co"] += 1

        # --- file video ---
        v = video.get(r.video_id)
        if v is not None:
            vid_path.append(str(v))
            dem["video_moi" if str(v) != str(r.video_path) else "video_giu"] += 1
        elif isinstance(r.video_path, str) and Path(r.video_path).exists():
            vid_path.append(r.video_path)
            dem["video_giu"] += 1
        else:
            vid_path.append(None)
            dem["video_khong_co"] += 1

    m["kf_path"], m["kf_name"], m["video_path"] = kf_path, kf_name, vid_path
    return m, dem


def kiem_toan_ven(cu: pd.DataFrame, moi: pd.DataFrame) -> list[str]:
    """Ba điều bất khả xâm phạm. Sai một điều là KHÔNG ĐƯỢC GHI."""
    loi = []
    if len(cu) != len(moi):
        loi.append(f"số dòng đổi: {len(cu):,} -> {len(moi):,}")
    if not cu.row_id.equals(moi.row_id):
        loi.append("cột `row_id` bị thay đổi — CẤM TUYỆT ĐỐI")
    duoc_sua = {"kf_path", "kf_name", "video_path"}
    for c in cu.columns:
        if c in duoc_sua:
            continue
        if not cu[c].equals(moi[c]):
            loi.append(f"cột `{c}` bị thay đổi ngoài ý muốn")
    return loi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--roots", nargs="*", default=MAC_DINH_ROOTS)
    ap.add_argument("--ghi", action="store_true", help="ghi thật (mặc định chỉ xem trước)")
    a = ap.parse_args()

    f = a.index / "master.parquet"
    cu = pd.read_parquet(f)
    print(f"{f}: {len(cu):,} dòng\n")

    print(f"Quét {len(a.roots)} nơi: {', '.join(str(x) for x in a.roots)}")
    thu_muc = quet_keyframe(a.roots)
    video = quet_video(a.roots)
    print(f"  {len(thu_muc)} thư mục keyframe, {len(video)} file .mp4\n")

    moi, dem = va(cu, thu_muc, video)

    loi = kiem_toan_ven(cu, moi)
    if loi:
        print("❌ KHÔNG GHI — phát hiện thay đổi ngoài ý muốn:")
        for x in loi:
            print("   ", x)
        raise SystemExit(1)

    print(f"{'':22} {'trước':>10} {'sau':>10} {'đổi':>10}")
    print("-" * 56)
    for ten, cot in (("keyframe có ảnh", "kf_path"), ("video có file", "video_path")):
        t = int(cu[cot].notna().sum())
        s = int(moi[cot].notna().sum())
        print(f"{ten:22} {t:>10,} {s:>10,} {s - t:>+10,}")
    print(f"\n  ảnh: {dem['anh_moi']:,} đổi đường dẫn, {dem['anh_giu']:,} giữ nguyên, "
          f"{dem['anh_khong_co']:,} chưa có")
    print(f"  video: {dem['video_moi']:,} đổi, {dem['video_giu']:,} giữ, "
          f"{dem['video_khong_co']:,} chưa có")

    nhom = (moi[moi.kf_path.notna()].video_id.str[:3].value_counts().sort_index())
    print(f"\nKeyframe có ảnh theo nhóm L: {dict(nhom)}")

    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi thật)")
        return

    sao_luu = f.with_suffix(".parquet.truoc_khi_va")
    shutil.copy2(f, sao_luu)
    moi.to_parquet(f, index=False)
    print(f"\n✅ Đã ghi {f}\n   sao lưu bản cũ: {sao_luu}")
    print("   `row_id` và mọi cột khác giữ nguyên — đã kiểm.")


if __name__ == "__main__":
    main()
