"""
01_build_index.py - Bước 2/3 của Giai đoạn 0.

Đọc index/paths.parquet, ghép 5 nguồn thành BẢNG CÁI + ma trận CLIP,
và báo cáo trung thực mọi chỗ không khớp.

Chạy:
    python 01_build_index.py --out ./index
    python 01_build_index.py --out ./index --limit 50     # thử nhanh 50 video

Output:
    index/master.parquet    một dòng = một keyframe
    index/clip.npy          (N, 512) float32, đã chuẩn hóa L2, cùng thứ tự row_id
    index/objects.parquet   detection >= ngưỡng, nối bằng row_id
    index/problems.csv      danh sách video có vấn đề
"""

import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png"}
NEED_COLS = {"n", "pts_time", "fps", "frame_idx"}


def load_meta(path):
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def build(paths: pd.DataFrame, limit=None):
    rows, problems, blocks = [], [], []
    row_id = 0
    if limit:
        paths = paths.head(limit)
    # parquet trả ô trống thành NaN (float) — mà NaN là truthy, nên
    # `if rec.keyframe_dir` sẽ lọt. Đổi hết về None trước khi dùng.
    paths = paths.astype(object).where(paths.notna(), None)

    for k, rec in enumerate(paths.itertuples(index=False), 1):
        vid = rec.video_id
        if k % 200 == 0:
            print(f"  ... {k}/{len(paths)} video")

        if not rec.csv:
            problems.append((vid, "thieu_csv", "")); continue
        df = pd.read_csv(rec.csv)
        if not NEED_COLS.issubset(df.columns):
            problems.append((vid, "csv_thieu_cot", str(list(df.columns)))); continue
        n_csv = len(df)

        # --- ảnh keyframe ---
        kf_files = []
        if rec.keyframe_dir and Path(rec.keyframe_dir).is_dir():
            kf_files = sorted(p for p in Path(rec.keyframe_dir).iterdir()
                              if p.suffix.lower() in IMG_EXT)
        if len(kf_files) != n_csv:
            problems.append((vid, "lech_so_keyframe", f"csv={n_csv} anh={len(kf_files)}"))

        # --- vector CLIP ---
        feat = None
        if rec.clip and Path(rec.clip).exists():
            feat = np.load(rec.clip).astype(np.float32)
            if feat.shape[0] != n_csv:
                problems.append((vid, "lech_so_vector", f"csv={n_csv} npy={feat.shape[0]}"))
                feat = None
            elif feat.shape[1] != 512:
                problems.append((vid, "sai_chieu_vector", str(feat.shape))); feat = None
        else:
            problems.append((vid, "thieu_npy", ""))

        # --- metadata video ---
        m = load_meta(rec.media) or {}
        if not m:
            problems.append((vid, "thieu_metadata", ""))

        # --- objects: khớp theo TÊN FILE keyframe, không theo chỉ số ---
        obj_dir = Path(rec.object_dir) if rec.object_dir else None
        obj_stems = set()
        if obj_dir and obj_dir.is_dir():
            obj_stems = {p.stem for p in obj_dir.glob("*.json")}

        for i in range(n_csv):
            r = df.iloc[i]
            kf = kf_files[i] if i < len(kf_files) else None
            # Quy ước: n=1 <-> 001.jpg <-> 001.json.
            # Ưu tiên khớp theo TÊN FILE ảnh (an toàn nhất). Khi chưa tải ảnh
            # keyframe thì vẫn khớp được object theo cột `n` — nếu không,
            # object của 96% video sẽ bị bỏ oan.
            stem = kf.stem if kf is not None else f"{int(r['n']):03d}"
            obj = str(obj_dir / f"{stem}.json") if stem in obj_stems else None

            rows.append({
                "row_id":       row_id,
                "video_id":     vid,
                "kf_n":         int(r["n"]),
                "kf_name":      kf.name if kf else None,
                "frame_idx":    int(r["frame_idx"]),     # <<< GIÁ TRỊ NỘP CHO BTC
                "pts_time":     float(r["pts_time"]),
                "fps":          float(r["fps"]),
                "kf_path":      str(kf) if kf else None,
                "obj_path":     obj,
                "video_path":   rec.video,
                "has_clip":     feat is not None,
                "title":        m.get("title", "") or "",
                "description":  m.get("description", "") or "",
                "keywords":     " ".join(m.get("keywords", []) or []),
                "video_length": m.get("length"),
                "publish_date": m.get("publish_date", "") or "",
            })
            row_id += 1

        blocks.append(feat if feat is not None
                      else np.zeros((n_csv, 512), dtype=np.float32))

    master = pd.DataFrame(rows)
    clip = np.vstack(blocks) if blocks else np.zeros((0, 512), np.float32)
    return master, clip, problems


def _read_one(args):
    """Đọc 1 file object JSON, trả về các detection qua ngưỡng."""
    row_id, p, min_score = args
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return []
    scores = d.get("detection_scores", [])
    names = d.get("detection_class_entities", [])
    boxes = d.get("detection_boxes", [])
    res = []
    for j, s in enumerate(scores):
        s = float(s)
        if s < min_score:
            break                       # mảng đã sắp giảm dần
        res.append({"row_id": row_id,
                    "label": names[j] if j < len(names) else None,
                    "score": s,
                    "box": boxes[j] if j < len(boxes) else None})
    return res


def build_objects(master: pd.DataFrame, min_score: float, workers: int = 16) -> pd.DataFrame:
    """Đọc ~180k file JSON nhỏ.

    Trên Windows đây là chỗ NGHẼN thật sự: mỗi lần mở file bị Defender quét,
    đọc tuần tự chỉ ~60 file/giây => gần 1 tiếng. Việc này nghẽn ở I/O chứ
    không phải CPU, nên đọc song song bằng thread ăn thua ngay (GIL được nhả
    trong lúc chờ đĩa). Muốn nhanh nữa thì loại trừ thư mục dữ liệu khỏi
    Windows Defender.
    """
    sub = master[master["obj_path"].notna()]
    tasks = [(r, p, min_score) for r, p in zip(sub["row_id"], sub["obj_path"])]
    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(_read_one, tasks, chunksize=64), 1):
            out.extend(res)
            if k % 20_000 == 0:
                sec = time.time() - t0
                print(f"  ... objects {k:,}/{len(tasks):,}  "
                      f"({k/sec:.0f} file/s, còn ~{(len(tasks)-k)/(k/sec)/60:.1f} phút)",
                      flush=True)
    print(f"  đọc xong {len(tasks):,} file trong {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("./index"), type=Path)
    ap.add_argument("--min-obj-score", default=0.3, type=float)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16,
                    help="số luồng đọc file object JSON")
    ap.add_argument("--objects-only", action="store_true",
                    help="dùng lại master.parquet/clip.npy đã có, chỉ dựng lại objects")
    a = ap.parse_args()

    if a.objects_only:
        print("Chế độ --objects-only: dùng lại master.parquet đã có.\n")
        master = pd.read_parquet(a.out / "master.parquet")
        clip = np.load(a.out / "clip.npy", mmap_mode="r")
        problems = []
    else:
        paths = pd.read_parquet(a.out / "paths.parquet")
        print(f"Đọc {len(paths):,} video từ paths.parquet\n")

        master, clip, problems = build(paths, a.limit)

        norm = np.linalg.norm(clip, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        clip = clip / norm

        master.to_parquet(a.out / "master.parquet", index=False)
        np.save(a.out / "clip.npy", clip)

    print("\nBóc tách objects ...")
    objects = build_objects(master, a.min_obj_score, a.workers)
    objects.to_parquet(a.out / "objects.parquet", index=False)

    # ---------------- BÁO CÁO ----------------
    L = []
    p = L.append
    p("=" * 64)
    p("BẢNG CÁI")
    p(f"  keyframe            {len(master):,}")
    p(f"  video               {master.video_id.nunique():,}")
    p(f"  ma trận CLIP        {clip.shape}   ({clip.nbytes/1e6:.0f} MB RAM)")
    p(f"  có ảnh keyframe     {master.kf_path.notna().mean()*100:5.1f}%")
    p(f"  có object JSON      {master.obj_path.notna().mean()*100:5.1f}%   <-- KIỂM TRA KỸ")
    p(f"  có metadata         {(master.title != '').mean()*100:5.1f}%")
    p(f"  có vector CLIP      {master.has_clip.mean()*100:5.1f}%")

    gaps = np.concatenate([np.diff(np.sort(g.values))
                           for _, g in master.groupby("video_id")["frame_idx"]
                           if len(g) > 1])
    p("")
    p("MẬT ĐỘ KEYFRAME  (quyết định có phải trích dày hay không)")
    p(f"  trung vị            {np.median(gaps):.0f} frame")
    p(f"  p90                 {np.percentile(gaps,90):.0f} frame")
    p(f"  % cặp cách <= 10    {(gaps<=10).mean()*100:.2f}%")
    p("  => " + ("XÁC NHẬN phải xây module trích dày cho TRAKE."
                 if np.median(gaps) > 10 else
                 "Keyframe đủ dày - xem lại giả định A1!"))

    p("")
    p(f"OBJECTS  (ngưỡng >= {a.min_obj_score})")
    if len(objects):
        p(f"  detection giữ lại   {len(objects):,}  "
          f"({len(objects)/max(len(master),1):.1f} / keyframe)")
        p("  top nhãn: " + ", ".join(objects.label.value_counts().head(15).index))
    else:
        p("  KHÔNG detection nào qua ngưỡng - kiểm tra lại object_dir.")

    p("")
    p(f"FPS xuất hiện: {sorted(master.fps.unique())}")
    d = master.groupby('video_id').pts_time.max().describe()
    p(f"Độ dài video (giây): min={d['min']:.0f}  trung vị={d['50%']:.0f}  max={d['max']:.0f}")

    p("")
    if problems:
        pr = pd.DataFrame(problems, columns=["video_id", "loai_loi", "chi_tiet"])
        pr.to_csv(a.out / "problems.csv", index=False)
        p(f"!! {len(problems)} vấn đề -> index/problems.csv")
        for k, v in pr.loai_loi.value_counts().items():
            p(f"   {k}: {v}")
    elif a.objects_only:
        p("(chế độ --objects-only: giữ nguyên index/problems.csv của lần chạy đầy đủ)")
    else:
        p("Không phát hiện vấn đề khớp dữ liệu nào.")
    p("")
    p("BƯỚC TIẾP: python 02_verify.py --out ./index --n 60")
    p("=" * 64)

    txt = "\n".join(L)
    print(txt)
    (a.out / "build_report.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
