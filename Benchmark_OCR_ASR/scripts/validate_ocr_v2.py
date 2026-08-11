from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from ocr_asr_benchmark.config import load_yaml
from ocr_asr_benchmark.utils.io import load_jsonl
from ocr_asr_benchmark.utils.text_metrics import normalize_text


def validate_labels(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    require_approved: bool,
) -> list[str]:
    errors: list[str] = []
    expected_positive = int(config['sample_policy']['positives_total'])
    expected_negative = int(config['sample_policy']['negatives_total'])
    if len(positives) != expected_positive:
        errors.append(f'Expected {expected_positive} positives, got {len(positives)}')
    if len(negatives) != expected_negative:
        errors.append(f'Expected {expected_negative} negatives, got {len(negatives)}')
    rows = positives + negatives
    if len({row.get('sample_id') for row in rows}) != len(rows):
        errors.append('OCR v2 sample IDs are not unique')
    if len({(row.get('video_id'), row.get('frame_idx')) for row in rows}) != len(rows):
        errors.append('OCR v2 frames are not unique')
    split_by_video: dict[str, set[str]] = {}
    normalized_by_split: dict[str, set[str]] = {}
    root = Path(config['dataset_root'])
    for row in rows:
        split_by_video.setdefault(row['video_id'], set()).add(row['split'])
        image = Path(row['image_path'])
        image = image if image.is_absolute() else root / image
        if not image.exists():
            errors.append(f'Missing image: {image}')
        if require_approved and (
            row.get('review_status') != 'approved' or row.get('second_review_status') != 'approved'
        ):
            errors.append(f'Not double-approved: {row["sample_id"]}')
    for video_id, splits in split_by_video.items():
        if len(splits) != 1:
            errors.append(f'Video leakage across splits: {video_id}')
    for row in positives:
        if row.get('label_kind') != 'positive':
            errors.append(f'Invalid positive kind: {row["sample_id"]}')
        bbox = row.get('bbox_xyxy')
        if not bbox or len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            errors.append(f'Invalid positive bbox: {row["sample_id"]}')
        normalized = normalize_text(row.get('text_raw', ''))
        if not normalized:
            errors.append(f'Empty positive text: {row["sample_id"]}')
        seen = normalized_by_split.setdefault(row['split'], set())
        if normalized in seen:
            errors.append(f'Duplicate normalized text in {row["split"]}: {normalized}')
        seen.add(normalized)
    for row in negatives:
        if row.get('label_kind') not in {'no_target_text', 'no_text_anywhere'}:
            errors.append(f'Invalid negative kind: {row["sample_id"]}')
    for domain in ('L21', 'L29'):
        for split in ('dev', 'holdout'):
            for name, collection in (('positive', positives), ('negative', negatives)):
                count = sum(row['domain'] == domain and row['split'] == split for row in collection)
                if count != 25:
                    errors.append(f'{domain}/{split}/{name}: expected 25, got {count}')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate OCR v2 labels and gates')
    parser.add_argument('--config', type=Path, default=BENCH_ROOT / 'configs/ocr_v2.yaml')
    parser.add_argument('--positive-labels', type=Path, default=BENCH_ROOT / 'eval_data/ocr_v2/positive_labels.jsonl')
    parser.add_argument('--negative-labels', type=Path, default=BENCH_ROOT / 'eval_data/ocr_v2/negative_labels.jsonl')
    parser.add_argument('--roi', type=Path, default=BENCH_ROOT / 'configs/roi_v2.yaml')
    parser.add_argument('--require-approved', action='store_true')
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args()
    config = load_yaml(args.config)
    missing = [path for path in (args.positive_labels, args.negative_labels) if not path.exists()]
    if missing:
        print('\n'.join(f'ERROR: Missing finalized label file: {path}' for path in missing))
        print('Run prepare_ocr_v2.py import, then finalize after double review.')
        return 1
    positives = load_jsonl(args.positive_labels)
    negatives = load_jsonl(args.negative_labels)
    errors = validate_labels(
        positives, negatives, config,
        require_approved=args.require_approved or args.require_ready,
    )
    if args.require_ready:
        roi = load_yaml(args.roi)
        for domain in ('L21', 'L29'):
            profile = roi.get('profiles', {}).get(domain, {})
            if profile.get('review_status') != 'approved' or not profile.get('include'):
                errors.append(f'ROI is not approved: {domain}')
        query_path = BENCH_ROOT / 'eval_data/ocr_v2/retrieval_queries.jsonl'
        queries = load_jsonl(query_path) if query_path.exists() else []
        if len(queries) != 40 or any(row.get('review_status') != 'approved' for row in queries):
            errors.append('Exactly 40 approved retrieval queries are required')
    if errors:
        print('\n'.join(f'ERROR: {error}' for error in errors))
        return 1
    print('OCR v2 validation passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
