from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from .config import BENCH_ROOT, load_yaml
from .safe_report import generate_report


def _parser(task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run each {} model in an isolated process'.format(task.upper()))
    parser.add_argument('--models', type=Path, default=BENCH_ROOT / 'configs/models.yaml')
    parser.add_argument('--benchmark', type=Path, default=BENCH_ROOT / 'configs/benchmark.yaml')
    default_labels = 'eval_data/ocr/ocr_labels.jsonl' if task == 'ocr' else 'eval_data/asr/transcripts.json'
    parser.add_argument('--labels', type=Path, default=BENCH_ROOT / default_labels)
    parser.add_argument('--output-dir', type=Path, default=BENCH_ROOT / 'results')
    parser.add_argument('--include-unreviewed', action='store_true')
    parser.add_argument('--limit', type=int)
    return parser


def _combine(task: str, output_dir: Path, part_dirs: list[Path]) -> None:
    for suffix in ('results', 'summary'):
        frames = []
        for directory in part_dirs:
            path = directory / '{}_{}.csv'.format(task, suffix)
            if path.exists() and path.stat().st_size:
                frames.append(pd.read_csv(path))
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        csv_path = output_dir / '{}_{}.csv'.format(task, suffix)
        combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
        if suffix == 'summary':
            records = combined.where(pd.notna(combined), None).to_dict(orient='records')
            (output_dir / '{}_summary.json'.format(task)).write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8'
            )


def run_isolated(task: str, args: argparse.Namespace) -> int:
    config = load_yaml(args.models)
    models = [item for item in config[task] if item.get('enabled', True)]
    part_root = args.output_dir / '.parts' / task
    part_root.mkdir(parents=True, exist_ok=True)
    part_dirs = []
    for model in models:
        part_dir = part_root / model['id']
        part_dir.mkdir(parents=True, exist_ok=True)
        part_dirs.append(part_dir)
        isolated_config = {
            'schema_version': config.get('schema_version', 1),
            'seed': config.get('seed', 2026),
            'device': config.get('device', 'cuda'),
            task: [model],
        }
        if task == 'asr':
            isolated_config['asr_generation'] = config['asr_generation']
        config_path = part_dir / 'models.yaml'
        config_path.write_text(
            yaml.safe_dump(isolated_config, allow_unicode=True, sort_keys=False), encoding='utf-8'
        )
        module = 'ocr_asr_benchmark.runners.{}'.format(task)
        command = [
            sys.executable, '-m', module, '--models', str(config_path),
            '--benchmark', str(args.benchmark), '--labels', str(args.labels),
            '--output-dir', str(part_dir),
        ]
        if args.include_unreviewed:
            command.append('--include-unreviewed')
        if args.limit is not None:
            command.extend(['--limit', str(args.limit)])
        print('Running isolated {} model: {}'.format(task.upper(), model['id']), flush=True)
        subprocess.run(command, cwd=BENCH_ROOT, check=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _combine(task, args.output_dir, part_dirs)
    return 0


def main_ocr() -> int:
    return run_isolated('ocr', _parser('ocr').parse_args())


def main_asr() -> int:
    return run_isolated('asr', _parser('asr').parse_args())


def main_all() -> int:
    parser = argparse.ArgumentParser(description='Run isolated OCR and ASR benchmarks')
    parser.add_argument('--include-unreviewed', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=BENCH_ROOT / 'results')
    args = parser.parse_args()
    common = ['--output-dir', str(args.output_dir)]
    if args.include_unreviewed:
        common.append('--include-unreviewed')
    run_isolated('ocr', _parser('ocr').parse_args(common))
    run_isolated('asr', _parser('asr').parse_args(common))
    print(generate_report(args.output_dir))
    return 0
