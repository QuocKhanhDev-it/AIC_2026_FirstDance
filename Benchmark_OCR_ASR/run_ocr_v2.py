from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ocr_asr_benchmark.config import BENCH_ROOT, load_yaml
from ocr_asr_benchmark.ocr_v2_report import generate_report
from ocr_asr_benchmark.utils.io import load_jsonl, write_csv, write_jsonl


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _models(config: dict[str, Any]) -> list[str]:
    return [item['id'] for item in config['recognizers'] if item.get('enabled', True)]


def _run(mode: str, model_id: str, args: argparse.Namespace, output: Path) -> None:
    command = [
        sys.executable, '-m', 'ocr_asr_benchmark.runners.ocr_v2', mode,
        '--model-id', model_id, '--config', str(args.config),
        '--roi', str(args.roi), '--positive-labels', str(args.positive_labels),
        '--negative-labels', str(args.negative_labels), '--output-dir', str(output),
    ]
    if args.include_unreviewed:
        command.append('--include-unreviewed')
    if args.limit is not None:
        command.extend(['--limit', str(args.limit)])
    print(f'Running OCR v2 {mode}: {model_id}', flush=True)
    subprocess.run(command, cwd=BENCH_ROOT, check=True)


def _combine_recognizer(output_dir: Path, model_ids: list[str]) -> None:
    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        part = output_dir / '.parts' / 'recognizer' / model_id
        jsonl_path = part / 'recognizer_results.jsonl'
        result_rows.extend(load_jsonl(jsonl_path) if jsonl_path.exists() else [])
        summary_rows.extend(_read_csv(part / 'recognizer_summary.csv'))
    write_csv(output_dir / 'recognizer_results.csv', result_rows)
    write_jsonl(output_dir / 'recognizer_results.jsonl', result_rows)
    write_csv(output_dir / 'recognizer_summary.csv', summary_rows)
    (output_dir / 'recognizer_summary.json').write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding='utf-8',
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Isolated OCR v2 benchmark orchestrator')
    parser.add_argument('mode', choices=['recognizer', 'gate', 'both'])
    parser.add_argument('--config', type=Path, default=BENCH_ROOT / 'configs/ocr_v2.yaml')
    parser.add_argument('--roi', type=Path, default=BENCH_ROOT / 'configs/roi_v2.yaml')
    parser.add_argument('--positive-labels', type=Path, default=BENCH_ROOT / 'eval_data/ocr_v2/positive_labels.jsonl')
    parser.add_argument('--negative-labels', type=Path, default=BENCH_ROOT / 'eval_data/ocr_v2/negative_labels.jsonl')
    parser.add_argument('--output-dir', type=Path, default=BENCH_ROOT / 'results/v2')
    parser.add_argument('--include-unreviewed', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    model_ids = _models(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {'recognizer', 'both'}:
        for model_id in model_ids:
            _run('recognizer', model_id, args, args.output_dir / '.parts' / 'recognizer' / model_id)
        _combine_recognizer(args.output_dir, model_ids)
        print(generate_report(args.output_dir))
    if args.mode in {'gate', 'both'}:
        for model_id in model_ids:
            _run('gate', model_id, args, args.output_dir / 'gate' / model_id)
        print(generate_report(args.output_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
