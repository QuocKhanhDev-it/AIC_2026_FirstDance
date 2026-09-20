"""
11_tim_cau_dem.py — Lọc ứng viên cho câu hỏi ĐẾM SỐ LƯỢNG.

Vì sao cần (A9): BTC nêu thẳng dạng câu này khi trả lời, và nói rõ vì sao nó
phá được cách chỉ dùng keyframe —

    "1 em bé được bế bởi 4 người liên tiếp trong bản tin, dựa trên keyframe
     thì chỉ có 3 nên không đảm bảo, dựa vào video thì đúng nhất"

Tập dev không có câu đếm thì **không bao giờ phát hiện được điểm yếu này**, và
cũng không đo được `trich_day` có đáng giữ không — vì đây là dạng câu duy nhất
nó thực sự cần thiết.

HAI LOẠI, và loại thứ hai mới là loại BTC nhấn mạnh:

    --khung   đếm vật trong MỘT khung   ("có bao nhiêu chiếc xe đạp?")
    --doan    đếm qua NHIỀU khung       ("bao nhiêu người bế em bé?")

    # xem dải frame trải đều qua một đoạn, để đếm bằng mắt
    python scripts/11_tim_cau_dem.py --dai L21_V024 5603 5610

⚠️ **Số của detector là CẬN DƯỚI, không phải đáp án.** D1.6 đo được: ảnh có
>20 người mà detector chỉ thấy 4. Vai trò của nó ở đây chỉ là **xếp thứ tự ưu
tiên** để người mở video đếm thật.

⚠️ **BÀI HỌC KHI DÙNG — số detector CAO là dấu hiệu XẤU.** Đã thử: khung có 10
`Boat` là bến cá hàng chục thuyền chồng lên nhau; khung có 8 `Flag` là hàng cờ
hút xa dần. **Không ai thống nhất được đáp án.** Câu đếm tốt cần số NHỎ (3–6)
và vật TÁCH BẠCH — nên mặc định lọc `--tu 3 --den 5`, không phải lấy top.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent

# Nhãn đếm được và dễ thống nhất đáp án. CỐ Ý BỎ `Person`, `Clothing`,
# `Human face`: cảnh đông người thì đếm bao nhiêu cũng cãi nhau.
DEM_DUOC = ["Boat", "Motorcycle", "Bicycle", "Car", "Truck", "Bus", "Canoe",
            "Bird", "Cattle", "Chair", "Bottle", "Bowl", "Balloon", "Flag",
            "Umbrella", "Dog", "Horse", "Tomato", "Fish", "Airplane"]


def _co_anh(m: pd.DataFrame, goc_anh=None) -> set:
    """row_id có ảnh trên máy này — kể cả nhóm L tải SAU khi dựng index.

    `master.parquet` ghi `kf_path` từ lúc dựng, nên nhóm tải sau vẫn hiện
    "chưa tải" dù ảnh đã nằm trên đĩa (A5.5). Quét thêm thư mục để bù.
    """
    ra = set(m[m.kf_path.notna()].row_id)
    for goc in (goc_anh or []):
        g = Path(goc)
        if not g.exists():
            continue
        for r in m.itertuples():
            if r.row_id in ra:
                continue
            if (g / r.video_id / f"{int(r.kf_n):03d}.jpg").exists():
                ra.add(r.row_id)
    return ra


def loc_khung(m, o, co_anh, bo_qua, tu=3, den=5) -> pd.DataFrame:
    """Khung có `tu`..`den` vật CÙNG MỘT LOẠI và không lẫn loại khác."""
    c = o[o.label.isin(DEM_DUOC) & o.row_id.isin(co_anh) & ~o.row_id.isin(bo_qua)]
    d = c.groupby(["row_id", "label"]).size().rename("so").reset_index()
    d = d[(d.so >= tu) & (d.so <= den)]
    d["tong_vat"] = d.row_id.map(c.groupby("row_id").size())
    d = d[d.tong_vat == d.so]          # chỉ một loại vật -> ít nhập nhằng
    d["video_id"] = d.row_id.map(m.video_id)
    d["kf"] = d.row_id.map(m.kf_n)
    return d.sort_values(["label", "so"])


def loc_doan(m, o, co_anh, toi_thieu_kf=5, toi_thieu_giay=5.0) -> pd.DataFrame:
    """Đoạn keyframe LIÊN TIẾP cùng chứa một nhãn — thường là một chuỗi sự kiện.

    Đây là nguồn cho câu đếm THEO THỜI GIAN, loại BTC nhấn mạnh.
    """
    c = o[o.label.isin(DEM_DUOC) & o.row_id.isin(co_anh)].copy()
    c["video_id"] = c.row_id.map(m.video_id)
    ra = []
    for (vid, nhan), g in c.groupby(["video_id", "label"]):
        r = np.sort(g.row_id.unique())
        for doan in np.split(r, np.where(np.diff(r) != 1)[0] + 1):
            if len(doan) < toi_thieu_kf:
                continue
            dai = float(m.pts_time.iloc[doan[-1]] - m.pts_time.iloc[doan[0]])
            if dai >= toi_thieu_giay:
                ra.append((vid, nhan, int(doan[0]), int(doan[-1]), len(doan), round(dai, 1)))
    return pd.DataFrame(ra, columns=["video_id", "label", "tu_row", "den_row",
                                     "so_kf", "dai_giay"]).sort_values(
        "so_kf", ascending=False)


def dung_dai(m, video_id: str, r0: int, r1: int, so_o: int = 16, ra_dir=None) -> Path | None:
    """Trích một dải frame trải đều qua đoạn, ghép thành MỘT ảnh để đếm bằng mắt.

    Đây đúng là công dụng mới của `trich_day` sau A9: **nhìn thấy nội dung**,
    không phải bắn trúng chỉ số.
    """
    import sys
    sys.path.insert(0, str(GOC / "src"))
    from PIL import Image, ImageDraw
    from trich_day import trich_day

    s = m[m.video_id == video_id].iloc[0]
    if not isinstance(s.video_path, str) or not Path(s.video_path).exists():
        print(f"{video_id}: không có file video trên máy này")
        return None

    a, b = m.iloc[r0], m.iloc[r1]
    fps = float(a.fps)
    ban_kinh = (float(b.pts_time) - float(a.pts_time)) / 2
    buoc = max(1, int(round(2 * ban_kinh * fps / (so_o - 1))))
    khung = trich_day(s.video_path, video_id,
                      (int(a.frame_idx) + int(b.frame_idx)) // 2,
                      (float(a.pts_time) + float(b.pts_time)) / 2,
                      fps, radius_sec=ban_kinh, stride=buoc, cache_dir=None)
    if not khung:
        return None

    C, W, H, NH = 4, 320, 180, 20
    hang = -(-len(khung) // C)
    sheet = Image.new("RGB", (W * C, (H + NH) * hang), "white")
    ve = ImageDraw.Draw(sheet)
    for i, k in enumerate(khung):
        x, y = (i % C) * W, (i // C) * (H + NH)
        im = k.anh.copy()
        im.thumbnail((W - 4, H - 4))
        sheet.paste(im, (x + 2, y + 2))
        ve.text((x + 4, y + H + 3), f"{k.pts_time:.1f}s  frame {k.frame_idx}",
                fill="#0645ad")
    f = Path(ra_dir or GOC / "dev" / "sheets") / f"DEM_{video_id}_{r0}_{r1}.jpg"
    f.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(f, quality=75)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--anh", nargs="*", default=[r"C:\Code\aic_data"],
                    help="thư mục gốc để quét ảnh của nhóm L tải sau")
    ap.add_argument("--khung", action="store_true", help="đếm trong một khung")
    ap.add_argument("--doan", action="store_true", help="đếm qua nhiều khung")
    ap.add_argument("--dai", nargs=3, metavar=("VIDEO_ID", "TU_ROW", "DEN_ROW"),
                    help="dựng dải frame của một đoạn để đếm bằng mắt")
    ap.add_argument("--tu", type=int, default=3)
    ap.add_argument("--den", type=int, default=5)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    m = pd.read_parquet(a.index / "master.parquet")

    if a.dai:
        f = dung_dai(m, a.dai[0], int(a.dai[1]), int(a.dai[2]))
        print(f"-> {f}" if f else "không dựng được")
        return

    o = pd.read_parquet(a.index / "objects.parquet")
    o = o[o.score >= 0.6]
    goc = [Path(x) / d for x in a.anh
           for d in ("Keyframes_L21/keyframes", "Keyframes_L22/keyframes",
                     "Keyframes_L27/keyframes")]
    co_anh = _co_anh(m, goc)

    import sys
    sys.path.insert(0, str(GOC / "src"))
    import tap_dev
    f = GOC / "dev" / "tap_dev.jsonl"
    bo = {x for c in tap_dev.doc(f) for x in
          (c.row_id_dung if not isinstance(c.row_id_dung[0], list)
           else [y for b in c.row_id_dung for y in b])} if f.exists() else set()

    print(f"{len(co_anh):,} keyframe có ảnh trên máy này\n")
    if a.khung or not a.doan:
        d = loc_khung(m, o, co_anh, bo, a.tu, a.den)
        print(f"=== ĐẾM TRONG MỘT KHUNG: {len(d)} ứng viên "
              f"({a.tu}–{a.den} vật, chỉ một loại) ===")
        print(d.head(a.n)[["row_id", "video_id", "kf", "label", "so"]].to_string(index=False))
    if a.doan:
        d = loc_doan(m, o, co_anh)
        print(f"\n=== ĐẾM QUA NHIỀU KHUNG: {len(d)} đoạn ===")
        print(d.head(a.n).to_string(index=False))
        print("\nDựng dải frame để đếm:  --dai <video_id> <tu_row> <den_row>")


if __name__ == "__main__":
    main()
