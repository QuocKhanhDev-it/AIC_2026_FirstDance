from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from ocr_asr_benchmark.utils.io import load_jsonl, write_jsonl

VALID_STATUS = {'needs_review', 'approved', 'rejected'}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def import_ocr(path: Path) -> None:
    edits = {row['sample_id']: row for row in read(path)}
    label_path = BENCH_ROOT / 'eval_data/ocr/ocr_labels.jsonl'
    labels = load_jsonl(label_path)
    if set(edits) != {row['sample_id'] for row in labels}:
        raise ValueError('OCR review CSV must contain every sample exactly once')
    for label in labels:
        edit = edits[label['sample_id']]
        if edit['review_status'] not in VALID_STATUS:
            raise ValueError('Invalid OCR review status: {}'.format(edit['review_status']))
        if label['text_type'] != 'no_text' and not edit['text_raw'].strip():
            raise ValueError('Approved OCR target cannot be empty: {}'.format(label['sample_id']))
        label['text_raw'] = edit['text_raw'].strip()
        label['text_type'] = edit['text_type'].strip()
        label['review_status'] = edit['review_status']
        label['notes'] = edit.get('notes', '').strip()
    write_jsonl(label_path, labels)


def import_asr(path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read(path):
        grouped[row['clip_id']].append(row)
    label_path = BENCH_ROOT / 'eval_data/asr/transcripts.json'
    document = json.loads(label_path.read_text(encoding='utf-8'))
    if set(grouped) != {clip['clip_id'] for clip in document['clips']}:
        raise ValueError('ASR review CSV must contain at least one row for every clip')
    for clip in document['clips']:
        edits = sorted(grouped[clip['clip_id']], key=lambda row: float(row['start']))
        statuses = {row['review_status'] for row in edits}
        if len(statuses) != 1 or not statuses <= VALID_STATUS:
            raise ValueError('Every segment in a clip must have one valid shared status')
        segments = []
        previous_end = 0.0
        for number, edit in enumerate(edits, start=1):
            start, end = float(edit['start']), float(edit['end'])
            if start < previous_end or end <= start or end > float(clip['duration_sec']) + 0.01:
                raise ValueError('Invalid timestamp in {}'.format(clip['clip_id']))
            if not edit['text_raw'].strip():
                raise ValueError('Empty transcript segment in {}'.format(clip['clip_id']))
            segments.append({
                'segment_id': edit.get('segment_id') or '{}_s{:02d}'.format(clip['clip_id'], number),
                'start': start, 'end': end, 'text_raw': edit['text_raw'].strip(),
                'reviewed_manually': next(iter(statuses)) == 'approved',
            })
            previous_end = end
        clip['segments'] = segments
        clip['audio_tags'] = [tag.strip() for tag in edits[0]['audio_tags'].split(',') if tag.strip()]
        clip['review_status'] = next(iter(statuses))
        clip['notes'] = edits[0].get('notes', '').strip()
    label_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Import manually reviewed OCR/ASR CSV files')
    parser.add_argument('--ocr', action='store_true')
    parser.add_argument('--asr', action='store_true')
    args = parser.parse_args()
    if not args.ocr and not args.asr:
        args.ocr = args.asr = True
    if args.ocr:
        import_ocr(BENCH_ROOT / 'eval_data/review/ocr_review.csv')
    if args.asr:
        import_asr(BENCH_ROOT / 'eval_data/review/asr_review.csv')
    print('Imported review CSV files; run validate_labels.py next')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
