"""
39_chon_dai_soan.py — Chọn DẢI keyframe để soạn câu dev, rồi dựng contact sheet.

    python scripts/39_chon_dai_soan.py --nhom L21 L22 L24 L27 L30 --so 30

VÌ SAO CÓ SCRIPT NÀY
====================

`10_contact_sheet.py` dựng sheet cho **một video** mình đã biết muốn xem. Nhưng
khi cần soạn HÀNG CHỤC câu thì việc tốn thời gian không phải dựng sheet — mà là
**chọn xem video nào**, và chọn sao cho không tự lừa mình:

* **Không được trùng video đã dùng làm đáp án.** Soạn thêm câu trên cùng video
  cũ thì tập dev "to ra" mà độ phủ không đổi.
* **Tránh đầu và cuối video.** Đó gần như luôn là hình hiệu, MC trong trường
  quay, hoặc danh sách chữ chạy — giống hệt nhau giữa hàng trăm bản tin, nên
  câu hỏi viết ra sẽ mơ hồ và đáp án không duy nhất.
* **Phải là DẢI liên tiếp, không phải một khung lẻ.** Đề thật tả một *đoạn*
  (A9: khoảng đúng dài 4 giây – 5 phút), nên câu dev cũng phải tả một đoạn.
  Tả một khung lẻ là quay lại đúng phân bố câu ngắn đã làm tập dev mù 6 lần.

Chọn theo hạt giống cố định (`--seed`) để chạy lại ra đúng danh sách cũ.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "scripts"))


def videos_chua_dung(master: pd.DataFrame, tap_dev: Path) -> set[str]:
    """Video đã là đáp án của một câu dev nào đó -> loại."""
    if not tap_dev.exists():
        return set()
    vid = master.video_id.values
    ra = set()
    for dong in tap_dev.read_text("utf-8").splitlines():
        if not dong.strip():
            continue
        for r in json.loads(dong)["row_id_dung"]:
            for x in (r if isinstance(r, list) else [r]):
                ra.add(str(vid[x]))
    return ra


def chon_dai(lat: pd.DataFrame, dai: int, rng: random.Random) -> pd.DataFrame:
    """Một dải `dai` keyframe liên tiếp, nằm trong khoảng giữa của video."""
    n = len(lat)
    if n < dai + 4:
        return lat.head(dai)
    lo, hi = int(n * 0.12), int(n * 0.88) - dai
    if hi <= lo:
        lo, hi = 0, n - dai
    i = rng.randint(lo, hi)
    return lat.iloc[i:i + dai]


def main():
    import importlib
    cs = importlib.import_module("10_contact_sheet")

    ap = argparse.ArgumentParser(description="chon dai keyframe de soan cau dev")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--tap-dev", default=GOC / "dev" / "tap_dev.jsonl", type=Path)
    ap.add_argument("--out", default=GOC / "dev" / ".sheet_soan", type=Path)
    ap.add_argument("--nhom", nargs="+", default=["L21", "L22", "L24", "L27", "L30"])
    ap.add_argument("--so", type=int, default=30, help="số dải cần chọn")
    ap.add_argument("--dai", type=int, default=10, help="số keyframe mỗi dải")
    ap.add_argument("--ca-video", action="store_true",
                    help="thay vì một dải, dựng sheet THƯA cả video (để soạn TRAKE)")
    ap.add_argument("--thua", type=int, default=10, help="dùng với --ca-video")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bo-qua", nargs="*", default=[],
                    help="video_id đã dựng rồi, không dựng lại")
    a = ap.parse_args()

    m = pd.read_parquet(a.index / "master.parquet")
    da_dung = videos_chua_dung(m, a.tap_dev) | set(a.bo_qua)
    co = m[m.kf_path.notna()]
    ung = sorted(v for v in co.video_id.unique()
                 if v[:3] in a.nhom and v not in da_dung)
    if not ung:
        raise SystemExit("khong con video nao chua dung trong cac nhom nay")

    rng = random.Random(a.seed)
    rng.shuffle(ung)
    chon = ung[:a.so]

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for v in sorted(chon):
        lat = co[co.video_id == v].sort_values("kf_n")
        if a.ca_video:
            phan = lat.iloc[::a.thua]
            ten = f"{v}_ca-video.jpg"
        else:
            phan = chon_dai(lat, a.dai, rng)
            ten = f"{v}_kf{int(phan.kf_n.iloc[0])}-{int(phan.kf_n.iloc[-1])}.jpg"
        t0, t1 = float(phan.pts_time.iloc[0]), float(phan.pts_time.iloc[-1])
        p = cs.dung_sheet(phan, a.out / ten, f"{v}  {t0:.0f}s - {t1:.0f}s")
        if p is None:
            continue
        manifest.append({
            "video_id": v, "sheet": str(p),
            "row_id": [int(x) for x in phan.row_id],
            "kf_n": [int(x) for x in phan.kf_n],
            "giay": [round(float(x), 1) for x in phan.pts_time],
        })
        print(f"{v:12} {t0:7.0f}s-{t1:<7.0f}s  {p.name}")

    (a.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    print(f"\n{len(manifest)} sheet -> {a.out}")


if __name__ == "__main__":
    main()
