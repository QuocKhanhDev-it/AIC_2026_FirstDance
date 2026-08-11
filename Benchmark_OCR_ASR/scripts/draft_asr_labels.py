from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from ocr_asr_benchmark.config import load_yaml
from ocr_asr_benchmark.asr_loader import load_asr_adapter


def save(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Create review-only ASR transcript drafts')
    parser.add_argument('--model-id', default='phowhisper_small')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    models = load_yaml(BENCH_ROOT / 'configs/models.yaml')
    model = next(item for item in models['asr'] if item['id'] == args.model_id)
    path = BENCH_ROOT / 'eval_data/asr/transcripts.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    adapter = load_asr_adapter(model, models['asr_generation'])

    drafted = 0
    for clip in document['clips'][:args.limit]:
        if clip.get('segments') and not args.force:
            continue
        prediction = adapter.predict((BENCH_ROOT / clip['audio_path']).resolve())
        segments = []
        for number, chunk in enumerate(prediction.get('chunks', []), start=1):
            text = str(chunk.get('text', '')).strip()
            timestamp = chunk.get('timestamp') or chunk.get('timestamps') or (None, None)
            start = 0.0 if timestamp[0] is None else max(0.0, float(timestamp[0]))
            end = float(clip['duration_sec']) if timestamp[1] is None else float(timestamp[1])
            end = min(float(clip['duration_sec']), end)
            if text and end > start:
                segments.append({
                    'segment_id': '{}_s{:02d}'.format(clip['clip_id'], number),
                    'start': round(start, 2), 'end': round(end, 2),
                    'text_raw': text, 'draft_model': model['repo_id'],
                })
        full_text = str(prediction.get('text', '')).strip()
        if not segments and full_text:
            segments = [{
                'segment_id': clip['clip_id'] + '_s01', 'start': 0.0,
                'end': float(clip['duration_sec']), 'text_raw': full_text,
                'draft_model': model['repo_id'],
            }]
        clip['segments'] = segments
        clip['audio_tags'] = ['speech'] if segments else ['no_speech']
        clip['review_status'] = 'needs_review'
        clip['notes'] = 'Model-assisted draft only; listen, correct, then approve'
        drafted += 1
        save(path, document)
        print('Drafted {} ({}/{})'.format(clip['clip_id'], drafted, len(document['clips'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
