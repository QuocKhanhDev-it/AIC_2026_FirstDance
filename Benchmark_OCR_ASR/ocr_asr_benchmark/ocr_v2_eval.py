from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .utils.text_metrics import corpus_wer, normalize_text, token_prf


def normalized_box(box: Sequence[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def box_in_roi(
    box: Sequence[float],
    width: int,
    height: int,
    include: Sequence[float],
    excludes: Iterable[Sequence[float]],
) -> bool:
    x1, y1, x2, y2 = normalized_box(box, width, height)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)

    def contains(region: Sequence[float]) -> bool:
        return region[0] <= center[0] <= region[2] and region[1] <= center[1] <= region[3]

    return contains(include) and not any(contains(region) for region in excludes)


def summarize_recognizer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get('status') == 'OK':
            groups[(row['model_id'], row['split'], row['domain'])].append(row)
            groups[(row['model_id'], row['split'], 'ALL')].append(row)
    summaries = []
    for (model_id, split, domain), items in sorted(groups.items()):
        pairs = [(row['reference'], row['prediction']) for row in items]
        summaries.append({
            'model_id': model_id,
            'split': split,
            'domain': domain,
            'samples': len(items),
            'corpus_wer_norm': corpus_wer(pairs),
            'macro_wer_norm': float(np.mean([row['wer_norm'] for row in items])),
            'exact_norm': float(np.mean([bool(row['exact_norm']) for row in items])),
            'token_precision': float(np.mean([row['token_precision'] for row in items])),
            'token_recall': float(np.mean([row['token_recall'] for row in items])),
            'token_f1': float(np.mean([row['token_f1'] for row in items])),
            'idf_token_recall': float(np.mean([row['idf_token_recall'] for row in items])),
            'cer_norm_diagnostic': float(np.mean([row['cer_norm'] for row in items])),
            'latency_median_sec': float(np.median([row['latency_sec'] for row in items])),
            'latency_p95_sec': float(np.percentile([row['latency_sec'] for row in items], 95)),
        })
    return summaries


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    positive = [row for row in rows if row['target_present']]
    negative = [row for row in rows if not row['target_present']]

    def accepted(row: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            candidate for candidate in row.get('candidates', [])
            if bool(normalize_text(candidate.get('text', '')))
            and float(candidate.get('confidence') or 0.0) >= threshold
        ]

    true_positive = sum(
        any(candidate.get('matched_target') for candidate in accepted(row))
        for row in positive
    )
    false_positive = sum(bool(accepted(row)) for row in negative)
    predicted_positive = sum(bool(accepted(row)) for row in rows)
    recall = true_positive / len(positive) if positive else 0.0
    fpr = false_positive / len(negative) if negative else 0.0
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'threshold': threshold,
        'positives': len(positive),
        'negatives': len(negative),
        'positive_recall': recall,
        'false_positive_rate': fpr,
        'precision': precision,
        'f1': f1,
    }


def choose_threshold(
    rows: list[dict[str, Any]],
    thresholds: Sequence[float],
    *,
    min_recall: float,
) -> tuple[float | None, list[dict[str, float]]]:
    curve = [threshold_metrics(rows, value) for value in thresholds]
    eligible = [row for row in curve if row['positive_recall'] >= min_recall]
    if not eligible:
        return None, curve
    winner = min(eligible, key=lambda row: (row['false_positive_rate'], -row['threshold']))
    return float(winner['threshold']), curve


def retrieval_tokens(
    reference: str, hypothesis: str, *, idf: dict[str, float] | None = None,
) -> dict[str, float]:
    output = token_prf(reference, hypothesis)
    output['idf_token_recall'] = token_prf(reference, hypothesis, idf=idf)['token_recall']
    return output


def paired_cluster_bootstrap(
    rows: list[dict[str, Any]],
    first_model: str,
    second_model: str,
    *,
    metric: str,
    iterations: int = 10000,
    seed: int = 2026,
) -> dict[str, float]:
    '''Paired bootstrap over videos; returns first minus second.'''
    relevant = [
        row for row in rows
        if row.get('status') == 'OK'
        and row.get('split') == 'holdout'
        and row.get('model_id') in {first_model, second_model}
    ]
    videos = sorted({row['video_id'] for row in relevant})
    by_key = {(row['model_id'], row['sample_id']): row for row in relevant}
    samples_by_video: dict[str, list[str]] = defaultdict(list)
    for row in relevant:
        if row['model_id'] == first_model:
            samples_by_video[row['video_id']].append(row['sample_id'])
    if not videos or any((second_model, sample) not in by_key for values in samples_by_video.values() for sample in values):
        raise ValueError('Models do not have paired holdout predictions')

    def score(model_id: str, sample_ids: list[str]) -> float:
        selected = [by_key[(model_id, sample_id)] for sample_id in sample_ids]
        if metric == 'corpus_wer_norm':
            return corpus_wer([(row['reference'], row['prediction']) for row in selected])
        if metric == 'exact_norm':
            return float(np.mean([bool(row['exact_norm']) for row in selected]))
        raise ValueError(f'Unsupported paired metric: {metric}')

    all_samples = [sample for video in videos for sample in samples_by_video[video]]
    observed = score(first_model, all_samples) - score(second_model, all_samples)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        drawn_videos = rng.choice(videos, size=len(videos), replace=True)
        sample_ids = [sample for video in drawn_videos for sample in samples_by_video[str(video)]]
        deltas.append(score(first_model, sample_ids) - score(second_model, sample_ids))
    return {
        'delta': observed,
        'ci_low': float(np.percentile(deltas, 2.5)),
        'ci_high': float(np.percentile(deltas, 97.5)),
    }
