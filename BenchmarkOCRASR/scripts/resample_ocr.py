from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))
DATASET_ROOT = Path("D:/Study/AICChallenge")
IGNORE_TEXT = {"online", "htv", "htv online", "hd", "htv9", "htv7", "9"}


def frame_map(video_id: str) -> dict[int, dict[str, str]]:
    path = DATASET_ROOT / "map-keyframes" / f"{video_id}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {int(row["n"]): row for row in csv.DictReader(stream)}


def predictions(reader: object, image_path: Path) -> list[tuple[float, list[float], str, float]]:
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    found = []
    for polygon, text, confidence in reader.readtext(image, detail=1, paragraph=False):
        clean = " ".join(str(text).split())
        normalized = clean.casefold().strip(" .:-_")
        xs, ys = [float(point[0]) for point in polygon], [float(point[1]) for point in polygon]
        bbox = [max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys))]
        in_logo_zone = bbox[0] < 180 and bbox[1] < 130
        if len(clean) < 3 or normalized in IGNORE_TEXT or in_logo_zone:
            continue
        area_ratio = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (width * height)
        score = float(confidence) * len(clean) * min(1.0, area_ratio * 180)
        found.append((score, bbox, clean, float(confidence)))
    return sorted(found, reverse=True, key=lambda item: item[0])


def choose_video(reader: object, video_id: str, sample_count: int) -> list[dict]:
    source_dir = DATASET_ROOT / "Keyframes_L29/keyframes" / video_id
    images = sorted(source_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    mapping = frame_map(video_id)
    edges = np.linspace(0, len(images), sample_count + 1, dtype=int)
    chosen, seen_text = [], set()
    for bucket, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        indexes = np.linspace(start, max(start, end - 1), min(12, max(1, end - start)), dtype=int)
        candidates = []
        for index in sorted(set(indexes.tolist())):
            for score, bbox, text, confidence in predictions(reader, images[index])[:4]:
                duplicate_penalty = 0.15 if text.casefold() in seen_text else 1.0
                candidates.append((score * duplicate_penalty, index, bbox, text, confidence))
        if candidates:
            score, index, bbox, text, confidence = max(candidates, key=lambda item: item[0])
        else:
            index = int((start + max(start, end - 1)) / 2)
            score, bbox, text, confidence = 0.0, [0, 0, 1280, 720], "", 0.0
        source = images[index]
        n = int(source.stem)
        row_map = mapping[n]
        destination = BENCH_ROOT / "eval_data/ocr/images" / f"{video_id}_{n:03d}.jpg"
        shutil.copy2(source, destination)
        vertical_center = (bbox[1] + bbox[3]) / 2
        chosen.append({
            "sample_id": f"ocr_{video_id}_{bucket:02d}", "video_id": video_id,
            "image_path": destination.relative_to(BENCH_ROOT).as_posix(), "kf_n": n,
            "frame_idx": int(row_map["frame_idx"]), "pts_time": float(row_map["pts_time"]),
            "text_type": "dynamic_overlay" if vertical_center >= 468 else "static_text",
            "bbox_xyxy": [round(float(value), 2) for value in bbox], "text_raw": text,
            "legibility": "auto_draft", "draft_confidence": confidence,
            "selection_score": score, "review_status": "needs_review",
            "notes": "Watermark-filtered EasyOCR draft; visually transcribe before approval",
        })
        if not text:
            chosen[-1]['text_type'] = 'no_text'
            chosen[-1]['legibility'] = 'no_text'
            chosen[-1]['notes'] = 'Intentional negative sample: no visible target text'
        if text:
            seen_text.add(text.casefold())
    return chosen


def contact_sheet(rows: list[dict], video_id: str) -> None:
    full_sheet = Image.new("RGB", (1600, 780), "white")
    crop_sheet = Image.new("RGB", (1600, 600), "white")
    for index, row in enumerate(rows):
        image = Image.open(BENCH_ROOT / row["image_path"]).convert("RGB")
        full = image.copy(); ImageDraw.Draw(full).rectangle(row["bbox_xyxy"], outline="red", width=5)
        full.thumbnail((400, 225)); tile = Image.new("RGB", (400, 260), "white"); tile.paste(full, (0, 0))
        ImageDraw.Draw(tile).text((4, 230), row["sample_id"], fill="black")
        full_sheet.paste(tile, ((index % 4) * 400, (index // 4) * 260))
        x1, y1, x2, y2 = row["bbox_xyxy"]
        pad = 12; crop = image.crop((max(0, x1-pad), max(0, y1-pad), min(image.width, x2+pad), min(image.height, y2+pad)))
        crop.thumbnail((380, 120)); crop_tile = Image.new("RGB", (400, 200), "white"); crop_tile.paste(crop, (10, 10))
        drawer = ImageDraw.Draw(crop_tile); drawer.text((5, 140), row["sample_id"], fill="black")
        drawer.text((5, 160), row["text_raw"][:55], fill="black")
        crop_sheet.paste(crop_tile, ((index % 4) * 400, (index // 4) * 200))
    output = BENCH_ROOT / "eval_data/ocr/contact_sheets"; output.mkdir(parents=True, exist_ok=True)
    full_sheet.save(output / f"{video_id}_full.jpg", quality=92)
    crop_sheet.save(output / f"{video_id}_crops.jpg", quality=94)


def main() -> int:
    easy_root = Path(os.environ["EASYOCR_MODULE_PATH"])
    if easy_root.drive.lower() != "d:": raise RuntimeError("EasyOCR cache must be on drive D")
    import easyocr
    reader = easyocr.Reader(["vi", "en"], gpu=True, model_storage_directory=str(easy_root / "model"), user_network_directory=str(easy_root / "user_network"))
    config = yaml.safe_load((BENCH_ROOT / "configs/benchmark.yaml").read_text(encoding="utf-8"))
    rows = []
    for video_id in config["selected_videos"]:
        selected = choose_video(reader, video_id, int(config["ocr_samples_per_video"]))
        rows.extend(selected); contact_sheet(selected, video_id); print(video_id, len(selected))
    path = BENCH_ROOT / "eval_data/ocr/ocr_labels.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} watermark-filtered OCR drafts")
    return 0


if __name__ == "__main__": raise SystemExit(main())
