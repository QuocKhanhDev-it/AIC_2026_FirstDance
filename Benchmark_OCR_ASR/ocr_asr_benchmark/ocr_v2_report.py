from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import BENCH_ROOT, load_yaml
from .ocr_v2_eval import paired_cluster_bootstrap
from .utils.io import load_jsonl


def select_recognizer(
    summary: pd.DataFrame,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    holdout = summary[(summary.split == 'holdout') & (summary.domain == 'ALL')].copy()
    if holdout.empty:
        return {'selected_model': None, 'reason': 'no holdout recognizer results'}
    ranked = holdout.sort_values(
        ['corpus_wer_norm', 'exact_norm', 'latency_median_sec'],
        ascending=[True, False, True],
    )
    first = str(ranked.iloc[0].model_id)
    if len(ranked) == 1:
        return {'selected_model': first, 'tied_models': [first]}
    second = str(ranked.iloc[1].model_id)
    iterations = int(config['bootstrap_iterations'])
    wer_delta = paired_cluster_bootstrap(
        rows, first, second, metric='corpus_wer_norm',
        iterations=iterations, seed=int(config['seed']),
    )
    exact_delta = paired_cluster_bootstrap(
        rows, first, second, metric='exact_norm',
        iterations=iterations, seed=int(config['seed']),
    )
    statistically_resolved = wer_delta['ci_high'] < 0 or (
        wer_delta['ci_low'] <= 0 <= wer_delta['ci_high'] and exact_delta['ci_low'] > 0
    )
    return {
        'selected_model': first if statistically_resolved else None,
        'provisional_model': first,
        'tied_models': [first] if statistically_resolved else [first, second],
        'wer_delta_first_minus_second': wer_delta,
        'exact_delta_first_minus_second': exact_delta,
        'reason': None if statistically_resolved else 'WER/Exact confidence intervals are inconclusive; retrieval must break the tie',
    }


def generate_report(output_dir: Path) -> Path:
    config = load_yaml(BENCH_ROOT / 'configs/ocr_v2.yaml')
    summary_path = output_dir / 'recognizer_summary.csv'
    results_path = output_dir / 'recognizer_results.jsonl'
    summary = pd.read_csv(summary_path) if summary_path.exists() and summary_path.stat().st_size else pd.DataFrame()
    rows = load_jsonl(results_path) if results_path.exists() else []
    selection = select_recognizer(summary, rows, config) if not summary.empty else {
        'selected_model': None, 'reason': 'no recognizer summary',
    }
    gate_summaries = {}
    gate_root = output_dir / 'gate'
    if gate_root.exists():
        for path in gate_root.glob('*/gate_summary.json'):
            gate_summaries[path.parent.name] = json.loads(path.read_text(encoding='utf-8'))
    conclusion = {
        'schema_version': 2,
        'recognizer': selection,
        'gate': gate_summaries,
        'status': 'awaiting_retrieval_ablation',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'conclusion.json').write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    table = '_No successful results._' if summary.empty else summary.round(4).to_markdown(index=False)
    report = f'''# OCR benchmark v2 — L21 + L29

Primary recognizer metric: normalized corpus WER. Exact match breaks statistical ties; CER is diagnostic only.

## Recognizer-only results

{table}

## Decision

```yaml
{yaml.safe_dump(conclusion, allow_unicode=True, sort_keys=False).strip()}
```
'''
    path = output_dir / 'summary_report.md'
    path.write_text(report, encoding='utf-8')
    return path
