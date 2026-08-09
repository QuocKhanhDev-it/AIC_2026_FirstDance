from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from ocr_asr_benchmark.utils.io import load_jsonl


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    review_dir = BENCH_ROOT / 'eval_data/review'
    ocr_rows = []
    for row in load_jsonl(BENCH_ROOT / 'eval_data/ocr/ocr_labels.jsonl'):
        ocr_rows.append({
            'sample_id': row['sample_id'], 'video_id': row['video_id'],
            'image_path': row['image_path'],
            'bbox_xyxy': json.dumps(row['bbox_xyxy']),
            'text_type': row['text_type'], 'text_raw': row['text_raw'],
            'review_status': row['review_status'], 'notes': row.get('notes', ''),
        })
    write(review_dir / 'ocr_review.csv', ocr_rows)

    document = json.loads((BENCH_ROOT / 'eval_data/asr/transcripts.json').read_text(encoding='utf-8'))
    asr_rows = []
    for clip in document['clips']:
        for segment in clip.get('segments', []):
            asr_rows.append({
                'clip_id': clip['clip_id'], 'video_id': clip['video_id'],
                'audio_path': clip['audio_path'],
                'segment_id': segment['segment_id'], 'start': segment['start'],
                'end': segment['end'], 'text_raw': segment['text_raw'],
                'audio_tags': ','.join(clip.get('audio_tags', [])),
                'review_status': clip['review_status'], 'notes': clip.get('notes', ''),
            })
    write(review_dir / 'asr_review.csv', asr_rows)
    print('Wrote review CSV files to {}'.format(review_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
