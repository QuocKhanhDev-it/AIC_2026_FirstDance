from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import BENCH_ROOT, load_yaml
from .report import _plots, _winner_ocr
from .utils.io import load_jsonl


def _approval_state() -> tuple[bool, bool]:
    ocr_labels = load_jsonl(BENCH_ROOT / 'eval_data/ocr/ocr_labels.jsonl')
    asr_document = json.loads((BENCH_ROOT / 'eval_data/asr/transcripts.json').read_text(encoding='utf-8'))
    return (
        bool(ocr_labels) and all(row.get('review_status') == 'approved' for row in ocr_labels),
        bool(asr_document['clips'])
        and all(row.get('review_status') == 'approved' for row in asr_document['clips']),
    )


def _winner_asr(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if frame.empty:
        return None, {'reason': 'no successful ASR results'}
    eligible = frame[
        (frame.timestamp_valid_rate >= config['asr_timestamp_validity_threshold'])
        & (frame.timestamp_coverage >= config['coverage_threshold'])
    ].sort_values(['wer_norm', 'rtf_median', 'vram_peak_gb'])
    if eligible.empty:
        return None, {'reason': 'no ASR model meets timestamp validity and coverage thresholds'}
    return str(eligible.iloc[0].model_id), {}


def generate_report(output_dir: Path = BENCH_ROOT / 'results') -> Path:
    config = load_yaml(BENCH_ROOT / 'configs/benchmark.yaml')
    ocr_path = output_dir / 'ocr_summary.csv'
    asr_path = output_dir / 'asr_summary.csv'
    ocr = pd.read_csv(ocr_path) if ocr_path.exists() and ocr_path.stat().st_size else pd.DataFrame()
    asr = pd.read_csv(asr_path) if asr_path.exists() and asr_path.stat().st_size else pd.DataFrame()
    ocr_approved, asr_approved = _approval_state()
    ocr_winner, ocr_notes = (
        _winner_ocr(ocr, config) if ocr_approved
        else (None, {'reason': 'OCR ground truth is not fully approved'})
    )
    asr_winner, asr_notes = (
        _winner_asr(asr, config) if asr_approved
        else (None, {'reason': 'ASR ground truth is not fully approved'})
    )
    _plots(ocr, asr, output_dir)

    def table(frame: pd.DataFrame) -> str:
        return '_No successful results._' if frame.empty else frame.round(4).to_markdown(index=False)

    conclusion = {
        'conclusion': {
            'ocr': {'selected_model': ocr_winner, **ocr_notes},
            'asr': {'selected_model': asr_winner, **asr_notes},
            'selection_policy': 'OCR: normalized WER, Exact, latency, VRAM; ASR: WER and timestamp gates',
            'ground_truth_approved': {'ocr': ocr_approved, 'asr': asr_approved},
        }
    }
    report = f'''# OCR & ASR Benchmark Summary — L29

Generated with Python {sys.version.split()[0]} on {platform.platform()}.

Ground truth approved: OCR={ocr_approved}, ASR={asr_approved}.

## OCR

{table(ocr)}

## ASR

{table(asr)}

## Machine-readable conclusion

```yaml
{yaml.safe_dump(conclusion, allow_unicode=True, sort_keys=False).strip()}
```
'''
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'summary_report.md'
    path.write_text(report, encoding='utf-8')
    (output_dir / 'conclusion.json').write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return path
