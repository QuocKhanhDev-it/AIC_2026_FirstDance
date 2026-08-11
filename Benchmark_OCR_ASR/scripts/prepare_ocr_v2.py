from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from ocr_asr_benchmark.config import load_yaml
from ocr_asr_benchmark.utils.io import load_jsonl, write_jsonl
from ocr_asr_benchmark.utils.text_metrics import normalize_text

CONFIG_PATH = BENCH_ROOT / 'configs/ocr_v2.yaml'
DATA_DIR = BENCH_ROOT / 'eval_data/ocr_v2'
CANDIDATES_PATH = DATA_DIR / 'candidates.jsonl'
REVIEW_PATH = DATA_DIR / 'review.csv'
POSITIVE_PATH = DATA_DIR / 'positive_labels.jsonl'
NEGATIVE_PATH = DATA_DIR / 'negative_labels.jsonl'
VALID_KINDS = {'positive', 'no_target_text', 'no_text_anywhere'}
SEMANTIC_TYPES = {'person_name', 'place', 'headline', 'number_date', 'other_static'}


def _frame_map(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {int(row['n']): row for row in csv.DictReader(stream)}


def _candidate_count(config: dict[str, Any], domain: str, split: str) -> int:
    per_bucket = config['sample_policy']['positives_total'] // 4
    return per_bucket * int(config['sample_policy']['candidate_multiplier'])


def make_contact_sheets(rows: list[dict[str, Any]], root: Path) -> None:
    output_dir = DATA_DIR / 'contact_sheets'
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row['video_id']].append(row)
    thumb_width, thumb_height, columns = 320, 200, 4
    for video_id, items in grouped.items():
        sheet = Image.new(
            'RGB', (thumb_width * columns, thumb_height * int(np.ceil(len(items) / columns))),
            'white',
        )
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(items):
            with Image.open(root / row['image_path']) as source:
                source_image = source.convert('RGB')
                source_width, source_height = source_image.size
                image = source_image.copy()
                image.thumbnail((thumb_width, thumb_height - 20))
            x = (index % columns) * thumb_width
            y = (index // columns) * thumb_height
            sheet.paste(image, (x, y + 20))
            draw.text((x + 4, y + 3), row['sample_id'], fill='black')
            scale_x, scale_y = image.width / source_width, image.height / source_height
            for prediction in row.get('draft_predictions', []):
                x1, y1, x2, y2 = prediction['bbox_xyxy']
                box = (x + x1 * scale_x, y + 20 + y1 * scale_y, x + x2 * scale_x, y + 20 + y2 * scale_y)
                draw.rectangle(box, outline='red', width=2)
        sheet.save(output_dir / f'{video_id}.jpg', quality=90)


def prepare(config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    root = Path(config['dataset_root'])
    rows: list[dict[str, Any]] = []
    for domain, domain_cfg in config['sample_policy']['domains'].items():
        for split in ('dev', 'holdout'):
            videos = domain_cfg[f'{split}_videos']
            target_count = _candidate_count(config, domain, split)
            base, remainder = divmod(target_count, len(videos))
            for video_number, video_id in enumerate(videos):
                count = base + (video_number < remainder)
                image_dir = root / domain_cfg['keyframe_root'] / video_id
                images = sorted(image_dir.glob('*.jpg'), key=lambda path: int(path.stem))
                if len(images) < count:
                    raise RuntimeError(f'Not enough keyframes for {video_id}: {len(images)} < {count}')
                mapping = _frame_map(root / 'map-keyframes' / f'{video_id}.csv')
                chosen = np.linspace(0, len(images) - 1, count, dtype=int)
                for image_index in chosen:
                    image = images[int(image_index)]
                    kf_n = int(image.stem)
                    mapped = mapping[kf_n]
                    rows.append({
                        'sample_id': f'ocrv2_{video_id}_{kf_n:04d}',
                        'domain': domain,
                        'split': split,
                        'video_id': video_id,
                        'kf_n': kf_n,
                        'frame_idx': int(mapped['frame_idx']),
                        'pts_time': float(mapped['pts_time']),
                        'image_path': image.relative_to(root).as_posix(),
                        'label_kind': 'needs_review',
                        'bbox_xyxy': None,
                        'text_raw': '',
                        'semantic_type': '',
                        'legibility': '',
                        'review_status': 'needs_review',
                        'second_review_status': 'needs_review',
                        'notes': '',
                        'draft_predictions': [],
                    })
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise RuntimeError('Candidate sample IDs are not unique')
    write_jsonl(CANDIDATES_PATH, rows)
    make_contact_sheets(rows, root)
    export_review()
    print(f'Prepared {len(rows)} time-stratified OCR v2 candidates')


def export_review() -> None:
    rows = load_jsonl(CANDIDATES_PATH)
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with REVIEW_PATH.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output['bbox_xyxy'] = json.dumps(row['bbox_xyxy']) if row['bbox_xyxy'] else ''
            output['draft_predictions'] = json.dumps(row.get('draft_predictions', []), ensure_ascii=False)
            writer.writerow(output)
    print(f'Wrote {REVIEW_PATH}')


def import_review() -> None:
    existing = {row['sample_id']: row for row in load_jsonl(CANDIDATES_PATH)}
    with REVIEW_PATH.open(encoding='utf-8-sig', newline='') as stream:
        edits = list(csv.DictReader(stream))
    if set(existing) != {row['sample_id'] for row in edits}:
        raise ValueError('Review CSV must contain every candidate exactly once')
    for edit in edits:
        row = existing[edit['sample_id']]
        for key in ('label_kind', 'text_raw', 'semantic_type', 'legibility', 'review_status', 'second_review_status', 'notes'):
            row[key] = edit.get(key, '').strip()
        bbox = edit.get('bbox_xyxy', '').strip()
        row['bbox_xyxy'] = json.loads(bbox) if bbox else None
    write_jsonl(CANDIDATES_PATH, existing.values())
    print(f'Imported {len(edits)} reviewed candidates')


def draft_predictions(config_path: Path = CONFIG_PATH) -> None:
    from ocr_asr_benchmark.model_loader import EasyOCRAdapter
    config = load_yaml(config_path)
    root = Path(config['dataset_root'])
    detector = EasyOCRAdapter(config['fixed_detector'])
    rows = load_jsonl(CANDIDATES_PATH)
    for index, row in enumerate(rows, start=1):
        predictions = detector.predict(root / row['image_path'])
        row['draft_predictions'] = [
            {
                'bbox_xyxy': [round(float(value), 2) for value in item['bbox_xyxy']],
                'text': item['text'], 'confidence': item['confidence'],
            }
            for item in predictions
        ]
        if index % 25 == 0:
            print(f'Drafted {index}/{len(rows)} frames', flush=True)
    write_jsonl(CANDIDATES_PATH, rows)
    make_contact_sheets(rows, root)
    export_review()


def _review_errors(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_frames: set[tuple[str, int]] = set()
    video_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = (row['video_id'], int(row['frame_idx']))
        if key in seen_frames:
            errors.append(f'Duplicate frame: {key}')
        seen_frames.add(key)
        video_splits[row['video_id']].add(row['split'])
        if row['review_status'] != 'approved' or row['second_review_status'] != 'approved':
            continue
        if row['label_kind'] not in VALID_KINDS:
            errors.append(f'Invalid label_kind: {row["sample_id"]}')
            continue
        if row['label_kind'] == 'positive':
            bbox = row.get('bbox_xyxy')
            if not bbox or len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                errors.append(f'Invalid positive bbox: {row["sample_id"]}')
            if not str(row.get('text_raw', '')).strip():
                errors.append(f'Empty positive text: {row["sample_id"]}')
            if row.get('semantic_type') not in SEMANTIC_TYPES:
                errors.append(f'Invalid semantic_type: {row["sample_id"]}')
    for video_id, splits in video_splits.items():
        if len(splits) > 1:
            errors.append(f'Video leakage across splits: {video_id}')
    return errors


def _round_robin(
    rows: list[dict[str, Any]],
    count: int,
    *,
    unique_text_seen: set[str] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: (item['video_id'], item['frame_idx'])):
        groups[row['video_id']].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(groups.values()):
        for video_id in sorted(groups):
            if groups[video_id] and len(selected) < count:
                candidate = groups[video_id].popleft()
                if unique_text_seen is not None:
                    normalized = normalize_text(candidate.get('text_raw', ''))
                    if normalized in unique_text_seen:
                        continue
                    unique_text_seen.add(normalized)
                selected.append(candidate)
    return selected


def finalize(config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    rows = load_jsonl(CANDIDATES_PATH)
    errors = _review_errors(rows)
    if errors:
        raise ValueError('\n'.join(errors))
    approved = [
        row for row in rows
        if row['review_status'] == 'approved' and row['second_review_status'] == 'approved'
    ]
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    seen_by_split: dict[str, set[str]] = defaultdict(set)
    quota = config['sample_policy']['positives_total'] // 4
    for domain in ('L21', 'L29'):
        for split in ('dev', 'holdout'):
            bucket = [row for row in approved if row['domain'] == domain and row['split'] == split]
            positive_bucket = [row for row in bucket if row['label_kind'] == 'positive']
            negative_bucket = [row for row in bucket if row['label_kind'] != 'positive']
            if len(positive_bucket) < quota or len(negative_bucket) < quota:
                raise RuntimeError(
                    f'{domain}/{split} needs {quota} positive and {quota} negative; '
                    f'got {len(positive_bucket)} and {len(negative_bucket)}'
                )
            selected_positive = _round_robin(
                positive_bucket, quota, unique_text_seen=seen_by_split[split],
            )
            if len(selected_positive) < quota:
                raise RuntimeError(f'{domain}/{split} has fewer than {quota} unique approved texts')
            positives.extend(selected_positive)
            negatives.extend(_round_robin(negative_bucket, quota))
    write_jsonl(POSITIVE_PATH, positives)
    write_jsonl(NEGATIVE_PATH, negatives)
    print(f'Finalized {len(positives)} positive and {len(negatives)} negative labels')


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare and review OCR benchmark v2 labels')
    parser.add_argument('action', choices=['prepare', 'draft', 'export', 'import', 'finalize'])
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.config)
    elif args.action == 'draft':
        draft_predictions(args.config)
    elif args.action == 'export':
        export_review()
    elif args.action == 'import':
        import_review()
    else:
        finalize(args.config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
