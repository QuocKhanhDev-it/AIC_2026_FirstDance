from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from ocr_asr_benchmark.config import load_yaml
from ocr_asr_benchmark.utils.io import load_jsonl, write_jsonl

CONFIG = BENCH_ROOT / 'configs/ocr_v2.yaml'
LABELS = BENCH_ROOT / 'eval_data/ocr_v2/positive_labels.jsonl'
QUERY_DIR = BENCH_ROOT / 'eval_data/ocr_v2'
QUERY_JSONL = QUERY_DIR / 'retrieval_queries.jsonl'
QUERY_CSV = QUERY_DIR / 'retrieval_queries_review.csv'
TYPE_QUOTAS = {'ocr_only': 8, 'mixed': 6, 'metadata_control': 6}


def _metadata(root: Path, video_id: str) -> dict:
    return json.loads((root / 'media-info' / f'{video_id}.json').read_text(encoding='utf-8'))


def prepare(config_path: Path = CONFIG, labels_path: Path = LABELS) -> None:
    config = load_yaml(config_path)
    root = Path(config['dataset_root'])
    labels = [row for row in load_jsonl(labels_path) if row['split'] == 'holdout']
    rows = []
    for domain in ('L21', 'L29'):
        domain_labels = [row for row in labels if row['domain'] == domain]
        if len(domain_labels) < TYPE_QUOTAS['ocr_only'] + TYPE_QUOTAS['mixed']:
            raise RuntimeError(f'Not enough holdout OCR labels for {domain}')
        for number, sample in enumerate(domain_labels[:TYPE_QUOTAS['ocr_only']], start=1):
            rows.append({
                'query_id': f'{domain.lower()}_ocr_{number:02d}', 'domain': domain,
                'query_type': 'ocr_only',
                'query': f'[REWRITE] Tìm cảnh có chữ {sample["text_raw"]}',
                'relevant_targets': [{'video_id': sample['video_id'], 'frame_idx': sample['frame_idx']}],
                'review_status': 'needs_review', 'notes': 'Rewrite without copying the OCR label verbatim',
            })
        start = TYPE_QUOTAS['ocr_only']
        mixed = domain_labels[start:start + TYPE_QUOTAS['mixed']]
        for number, sample in enumerate(mixed, start=1):
            title = _metadata(root, sample['video_id']).get('title', '')
            rows.append({
                'query_id': f'{domain.lower()}_mixed_{number:02d}', 'domain': domain,
                'query_type': 'mixed',
                'query': f'[REWRITE] Trong {title}, tìm cảnh liên quan {sample["text_raw"]}',
                'relevant_targets': [{'video_id': sample['video_id'], 'frame_idx': sample['frame_idx']}],
                'review_status': 'needs_review', 'notes': 'Rewrite before approval',
            })
        videos = list(dict.fromkeys(row['video_id'] for row in domain_labels))
        for number in range(TYPE_QUOTAS['metadata_control']):
            video_id = videos[number % len(videos)]
            sample = next(row for row in domain_labels if row['video_id'] == video_id)
            title = _metadata(root, video_id).get('title', '')
            rows.append({
                'query_id': f'{domain.lower()}_meta_{number + 1:02d}', 'domain': domain,
                'query_type': 'metadata_control', 'query': f'[REWRITE] {title}',
                'relevant_targets': [{'video_id': sample['video_id']}],
                'review_status': 'needs_review', 'notes': 'Choose an answerable metadata query and verify target',
            })
    write_jsonl(QUERY_JSONL, rows)
    export_review()
    print(f'Prepared {len(rows)} retrieval query drafts')


def export_review() -> None:
    rows = load_jsonl(QUERY_JSONL)
    QUERY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with QUERY_CSV.open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['query_id', 'domain', 'query_type', 'query', 'relevant_targets', 'review_status', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, relevant_targets=json.dumps(row['relevant_targets'], ensure_ascii=False)))


def import_review() -> None:
    with QUERY_CSV.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        row['relevant_targets'] = json.loads(row['relevant_targets'])
        output.append(row)
    write_jsonl(QUERY_JSONL, output)
    print(f'Imported {len(output)} reviewed queries')


def validate() -> None:
    rows = load_jsonl(QUERY_JSONL)
    errors = []
    if len(rows) != 40:
        errors.append(f'Expected 40 queries, got {len(rows)}')
    if len({row['query_id'] for row in rows}) != len(rows):
        errors.append('Query IDs are not unique')
    for domain in ('L21', 'L29'):
        for query_type, quota in TYPE_QUOTAS.items():
            count = sum(row['domain'] == domain and row['query_type'] == query_type for row in rows)
            if count != quota:
                errors.append(f'{domain}/{query_type}: expected {quota}, got {count}')
    for row in rows:
        if row.get('review_status') != 'approved':
            errors.append(f'Query is not approved: {row["query_id"]}')
        if '[REWRITE]' in row.get('query', '') or not row.get('query', '').strip():
            errors.append(f'Query still needs rewrite: {row["query_id"]}')
        if not row.get('relevant_targets'):
            errors.append(f'Query has no relevant target: {row["query_id"]}')
    if errors:
        raise ValueError('\n'.join(errors))
    print('OCR retrieval queries are valid and approved')


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare and review OCR retrieval queries')
    parser.add_argument('action', choices=['prepare', 'export', 'import', 'validate'])
    args = parser.parse_args()
    {'prepare': prepare, 'export': export_review, 'import': import_review, 'validate': validate}[args.action]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
