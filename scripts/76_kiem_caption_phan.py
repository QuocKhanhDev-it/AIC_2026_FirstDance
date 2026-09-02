"""
76_kiem_caption_phan.py — Soát các phần caption nhận về TRƯỚC khi ghép.

    python scripts/76_kiem_caption_phan.py
    python scripts/76_kiem_caption_phan.py --ghep      # ghép sau khi soát sạch

NĂM THỨ PHẢI ĐÚNG, cả năm đều hỏng ÂM THẦM nếu sai

**1. Đúng phần được giao.** Ai đó quên đổi `PHAN` thì nộp về caption của phần
người khác — file hợp lệ, tên đúng, nội dung sai. Chỉ lộ ra khi đối chiếu
`row_id` với `chia_caption/phan_N.txt`.

**2. Không trùng nhau giữa các phần.** Trùng thì ghép xong một số ảnh có hai
caption; `--bien` giữ bản cuối, tức kết quả phụ thuộc THỨ TỰ ghép.

**3. Cùng cấu hình sinh.** `--so-chu` khác nhau giữa các phần thì caption dài
ngắn khác nhau, BM25 chấm lệch theo phần — và không ai nhìn ra từ kết quả.
Dấu hiệu: phân bố độ dài lệch hẳn.

**4. Không có caption rỗng.** Ảnh hỏng hoặc model trả rỗng -> dòng vô dụng.

**5. Đủ số ảnh.** Phiên Kaggle chết giữa chừng thì `.jsonl` vẫn hợp lệ, chỉ
thiếu phần đuôi — nhìn số dòng mới biết.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))


def doc_phan(f: Path) -> pd.DataFrame:
    """Đọc .parquet, hoặc .jsonl/.txt (mỗi dòng một JSON)."""
    if f.suffix == ".parquet":
        return pd.read_parquet(f)
    d = [json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip()]
    return pd.DataFrame(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--thu-muc", default=GOC / "index" / "caption", type=Path)
    ap.add_argument("--chia", default=GOC / "chia_caption", type=Path)
    ap.add_argument("--ghep", action="store_true",
                    help="ghép vào index/caption.parquet (chỉ khi soát sạch)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    vid_theo_row = master.set_index("row_id").video_id

    tep = sorted(a.thu_muc.glob("caption_phan*.parquet"))
    if not tep:
        raise SystemExit(f"không thấy caption_phan*.parquet trong {a.thu_muc}")

    print(f"{'phần':<6}{'dòng':>8}{'video':>7}{'ĐÚNG phần':>11}"
          f"{'rỗng':>6}{'ký tự trung vị':>16}{'thiếu so với giao':>19}")
    print("-" * 74)

    tat_ca, loi = {}, []
    for f in tep:
        so = int(f.stem.replace("caption_phan", ""))
        d = doc_phan(f)
        cot = "caption" if "caption" in d.columns else d.columns[-1]
        d = d[["row_id", cot]].rename(columns={cot: "caption"})
        d["row_id"] = d.row_id.astype(int)

        giao = {x.strip() for x in
                (a.chia / f"phan_{so}.txt").read_text("utf-8").splitlines()
                if x.strip()}
        vid = set(vid_theo_row.loc[d.row_id])
        ngoai = vid - giao
        n_giao = int(master.video_id.isin(giao).sum())
        rong = int(d.caption.fillna("").str.strip().eq("").sum())
        dai = int(d.caption.str.len().median())

        print(f"{so:<6}{len(d):>8,}{len(vid):>7}"
              f"{'✅' if not ngoai else '❌ ' + str(len(ngoai)):>11}"
              f"{rong:>6}{dai:>16}{n_giao - len(d):>19,}")
        if ngoai:
            loi.append(f"phần {so}: {len(ngoai)} video KHÔNG thuộc phần được "
                       f"giao, ví dụ {sorted(ngoai)[:3]} — chạy nhầm PHAN?")
        if rong:
            loi.append(f"phần {so}: {rong} caption rỗng")
        if n_giao - len(d) > n_giao * 0.02:
            loi.append(f"phần {so}: thiếu {n_giao - len(d):,}/{n_giao:,} ảnh "
                       f"— phiên chết giữa chừng?")
        tat_ca[so] = d

    print()
    khoa = sorted(tat_ca)
    for i, x in enumerate(khoa):
        for y in khoa[i + 1:]:
            chung = set(tat_ca[x].row_id) & set(tat_ca[y].row_id)
            if chung:
                loi.append(f"phần {x} và {y} TRÙNG {len(chung):,} ảnh")

    # Cấu hình sinh có giống nhau không — đọc qua phân bố độ dài.
    dai = {k: v.caption.str.len().median() for k, v in tat_ca.items()}
    lo, hi = min(dai.values()), max(dai.values())
    print(f"độ dài caption trung vị: {lo:.0f}–{hi:.0f} ký tự "
          f"(lệch {(hi - lo) / lo * 100:.0f}%)")
    if hi > lo * 1.4:
        loi.append("phân bố độ dài lệch >40% — các phần có thể chạy KHÁC "
                   "`--so-chu`, BM25 sẽ chấm lệch theo phần")

    if loi:
        print("\n❌ CÓ VẤN ĐỀ:")
        for x in loi:
            print("   •", x)
        raise SystemExit(1)
    print("\n✅ Mọi phần đều sạch.")

    if not a.ghep:
        print("   (thêm --ghep để gộp vào index/caption.parquet)")
        return

    cu = a.index / "caption.parquet"
    gop = [pd.read_parquet(cu)] if cu.exists() else []
    gop += list(tat_ca.values())
    d = pd.concat(gop, ignore_index=True).drop_duplicates("row_id", keep="last")
    if cu.exists():
        cu.rename(cu.with_suffix(".parquet.truoc_khi_ghep"))
    d.to_parquet(cu, index=False)
    print(f"\n✅ {cu}: {len(d):,} ảnh "
          f"({vid_theo_row.loc[d.row_id].nunique()} video)")


if __name__ == "__main__":
    main()
