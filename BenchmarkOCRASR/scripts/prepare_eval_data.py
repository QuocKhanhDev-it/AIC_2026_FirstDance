from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_frame_map(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {int(row["n"]): row for row in csv.DictReader(stream)}


def candidate_indices(count: int, buckets: int, candidates_per_bucket: int = 4) -> list[list[int]]:
    edges = np.linspace(0, count, buckets + 1, dtype=int)
    result = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            result.append([min(start, count - 1)])
        else:
            result.append(sorted(set(np.linspace(start, end - 1, min(candidates_per_bucket, end - start), dtype=int).tolist())))
    return result


def best_text_prediction(reader: Any, image_path: Path) -> tuple[float, list[float], str, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0, [0, 0, 1, 1], "", 0.0
    height, width = image.shape[:2]
    outputs = reader.readtext(image, detail=1, paragraph=False)
    candidates = []
    for polygon, text, confidence in outputs:
        clean = " ".join(str(text).split())
        if len(clean) < 3:
            continue
        xs, ys = [float(p[0]) for p in polygon], [float(p[1]) for p in polygon]
        bbox = [max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys))]
        area_ratio = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (width * height)
        score = float(confidence) * max(3, len(clean)) * min(1.0, area_ratio * 120)
        if clean.upper() in {"HTV", "HTV9", "HTV7", "HD"}:
            score *= 0.05
        candidates.append((score, bbox, clean, float(confidence)))
    return max(candidates, default=(0.0, [0, 0, width, height], "", 0.0), key=lambda item: item[0])


def prepare_ocr(config: dict[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    easy_root = Path(os.environ.get("EASYOCR_MODULE_PATH", ""))
    if not easy_root or easy_root.drive.lower() != "d:":
        raise RuntimeError("EASYOCR_MODULE_PATH must be configured on drive D")
    import easyocr
    reader = easyocr.Reader(
        ["vi", "en"], gpu=True,
        model_storage_directory=str(easy_root / "model"),
        user_network_directory=str(easy_root / "user_network"),
    )

    output_images = BENCH_ROOT / "eval_data/ocr/images"
    output_images.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for video_id in config["selected_videos"]:
        source_dir = dataset_root / "Keyframes_L29/keyframes" / video_id
        images = sorted(source_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        frame_map = load_frame_map(dataset_root / "map-keyframes" / f"{video_id}.csv")
        buckets = candidate_indices(len(images), int(config["ocr_samples_per_video"]))
        for sample_number, indices in enumerate(buckets, start=1):
            scored = []
            for index in indices:
                score, bbox, text, confidence = best_text_prediction(reader, images[index])
                scored.append((score, index, bbox, text, confidence))
            score, index, bbox, text, confidence = max(scored, key=lambda item: item[0])
            source = images[index]
            n = int(source.stem)
            mapping = frame_map[n]
            destination = output_images / f"{video_id}_{n:03d}.jpg"
            shutil.copy2(source, destination)
            with Image.open(destination) as image:
                height = image.height
            vertical_center = (bbox[1] + bbox[3]) / 2
            text_type = "dynamic_overlay" if vertical_center >= height * 0.65 else "static_text"
            rows.append({
                "sample_id": f"ocr_{video_id}_{sample_number:02d}",
                "video_id": video_id,
                "image_path": destination.relative_to(BENCH_ROOT).as_posix(),
                "kf_n": n,
                "frame_idx": int(mapping["frame_idx"]),
                "pts_time": float(mapping["pts_time"]),
                "text_type": text_type,
                "bbox_xyxy": [round(float(value), 2) for value in bbox],
                "text_raw": text,
                "legibility": "auto_draft",
                "draft_confidence": confidence,
                "selection_score": score,
                "review_status": "needs_review",
                "notes": "EasyOCR-assisted draft; visually transcribe before approval",
            })
    return rows


def make_contact_sheets(rows: list[dict[str, Any]]) -> None:
    contact_dir = BENCH_ROOT / "eval_data/ocr/contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    for video_id in sorted({row["video_id"] for row in rows}):
        selected = [row for row in rows if row["video_id"] == video_id]
        thumbs = []
        for row in selected:
            image = Image.open(BENCH_ROOT / row["image_path"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle(row["bbox_xyxy"], outline="red", width=4)
            image.thumbnail((400, 225))
            canvas = Image.new("RGB", (400, 260), "white")
            canvas.paste(image, (0, 0))
            ImageDraw.Draw(canvas).text((4, 230), row["sample_id"], fill="black")
            thumbs.append(canvas)
        sheet = Image.new("RGB", (1600, 780), "white")
        for index, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((index % 4) * 400, (index // 4) * 260))
        sheet.save(contact_dir / f"{video_id}.jpg", quality=92)


def prepare_asr(config: dict[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    output_dir = BENCH_ROOT / "eval_data/asr/clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for video_id in config["selected_videos"]:
        video = dataset_root / "Videos_L29_a/video" / f"{video_id}.mp4"
        for clip_number, start in enumerate(config["asr_clip_starts_sec"], start=1):
            clip_id = f"asr_{video_id}_{clip_number:02d}"
            destination = output_dir / f"{clip_id}.wav"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(start), "-i", str(video), "-t", str(config["asr_clip_duration_sec"]),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(destination),
            ]
            subprocess.run(command, check=True)
            rows.append({
                "clip_id": clip_id,
                "video_id": video_id,
                "audio_path": destination.relative_to(BENCH_ROOT).as_posix(),
                "source_start_sec": float(start),
                "duration_sec": float(config["asr_clip_duration_sec"]),
                "segments": [],
                "audio_tags": [],
                "review_status": "needs_review",
                "notes": "Listen and add sentence-level segments before approval",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic L29 OCR/ASR labeling set")
    parser.add_argument("--config", type=Path, default=BENCH_ROOT / "configs/benchmark.yaml")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--asr", action="store_true")
    args = parser.parse_args()
    if not args.ocr and not args.asr:
        args.ocr = args.asr = True
    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    if args.ocr:
        ocr_rows = prepare_ocr(config, dataset_root)
        from ocr_asr_benchmark.utils.io import write_jsonl
        write_jsonl(BENCH_ROOT / "eval_data/ocr/ocr_labels.jsonl", ocr_rows)
        make_contact_sheets(ocr_rows)
        print(f"Prepared {len(ocr_rows)} OCR drafts")
    if args.asr:
        asr_rows = prepare_asr(config, dataset_root)
        path = BENCH_ROOT / "eval_data/asr/transcripts.json"
        path.write_text(json.dumps({"schema_version": 1, "clips": asr_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prepared {len(asr_rows)} ASR clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
