from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import soundfile as sf

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))


def validate_ocr(path: Path, require_approved: bool) -> list[str]:
    from ocr_asr_benchmark.utils.io import load_jsonl
    errors = []
    rows = load_jsonl(path)
    if len(rows) != 60:
        errors.append(f"OCR count must be 60, got {len(rows)}")
    if len({row.get("sample_id") for row in rows}) != len(rows):
        errors.append("OCR sample_id values are not unique")
    if len({(row.get("video_id"), row.get("frame_idx")) for row in rows}) != len(rows):
        errors.append("OCR video_id/frame_idx pairs are not unique")
    for row in rows:
        if row.get('text_type') == 'no_text':
            image = BENCH_ROOT / row['image_path']
            if not image.exists(): errors.append(f'Missing image: {image}')
            bbox = row.get('bbox_xyxy', [])
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                errors.append('Invalid bbox: {}'.format(row['sample_id']))
            if require_approved and row.get('review_status') != 'approved':
                errors.append('Not approved: {}'.format(row['sample_id']))
            continue
        image = BENCH_ROOT / row["image_path"]
        if not image.exists(): errors.append(f"Missing image: {image}")
        if not str(row.get("text_raw", "")).strip(): errors.append(f"Empty OCR text: {row['sample_id']}")
        bbox = row.get("bbox_xyxy", [])
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]: errors.append(f"Invalid bbox: {row['sample_id']}")
        if require_approved and row.get("review_status") != "approved": errors.append(f"Not approved: {row['sample_id']}")
    return errors


def validate_asr(path: Path, require_approved: bool) -> list[str]:
    errors = []
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document["clips"]
    if len(rows) != 20: errors.append(f"ASR count must be 20, got {len(rows)}")
    for row in rows:
        audio = BENCH_ROOT / row["audio_path"]
        if not audio.exists():
            errors.append(f"Missing audio: {audio}"); continue
        info = sf.info(audio)
        if info.samplerate != 16000 or info.channels != 1: errors.append(f"Invalid WAV format: {row['clip_id']}")
        if not row.get("segments"): errors.append(f"No ASR segments: {row['clip_id']}")
        previous_end = 0.0
        for segment in row.get("segments", []):
            start, end = float(segment["start"]), float(segment["end"])
            if start < previous_end or end <= start or end > float(row["duration_sec"]) + 0.01:
                errors.append(f"Invalid segment timestamp: {row['clip_id']}")
            if not str(segment.get("text_raw", "")).strip(): errors.append(f"Empty segment text: {row['clip_id']}")
            previous_end = end
        if require_approved and row.get("review_status") != "approved": errors.append(f"Not approved: {row['clip_id']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args()
    errors = validate_ocr(BENCH_ROOT / "eval_data/ocr/ocr_labels.jsonl", args.require_approved)
    errors += validate_asr(BENCH_ROOT / "eval_data/asr/transcripts.json", args.require_approved)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print("Label validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
