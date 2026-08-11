from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from .utils.text_metrics import tokenize_bm25


def rank_scores(scores: Sequence[float], *, positive_only: bool = False) -> dict[int, int]:
    values = np.asarray(scores, dtype=float)
    order = np.argsort(-values, kind='stable')
    if positive_only:
        order = order[values[order] > 0]
    return {int(document): rank for rank, document in enumerate(order, start=1)}


def reciprocal_rank_fusion(
    metadata_scores: Sequence[float],
    ocr_scores: Sequence[float],
    *,
    k: int = 60,
) -> np.ndarray:
    metadata_ranks = rank_scores(metadata_scores)
    ocr_ranks = rank_scores(ocr_scores, positive_only=True)
    fused = np.zeros(len(metadata_scores), dtype=float)
    for document, rank in metadata_ranks.items():
        fused[document] += 1.0 / (k + rank)
    for document, rank in ocr_ranks.items():
        fused[document] += 1.0 / (k + rank)
    return fused


def first_relevant_rank(order: Sequence[int], relevant: set[int]) -> int | None:
    for rank, document in enumerate(order, start=1):
        if int(document) in relevant:
            return rank
    return None


def query_metrics(first_rank: int | None, ranks: Sequence[int]) -> dict[str, float | int | None]:
    output: dict[str, float | int | None] = {
        'first_relevant_rank': first_rank,
        'mrr': 0.0 if first_rank is None else 1.0 / first_rank,
    }
    recalls = []
    for rank in ranks:
        hit = float(first_rank is not None and first_rank <= rank)
        output[f'recall_at_{rank}'] = hit
        recalls.append(hit)
    output['final_score'] = float(np.mean(recalls))
    return output


def summarize_queries(rows: list[dict[str, Any]], ranks: Sequence[int]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for domain in (row['domain'], 'ALL'):
            for query_type in (row['query_type'], 'ALL'):
                groups[(row['system'], domain, query_type)].append(row)
    summaries = []
    for (system, domain, query_type), items in sorted(groups.items()):
        output = {
            'system': system, 'domain': domain, 'query_type': query_type,
            'queries': len(items),
            'final_score': float(np.mean([row['final_score'] for row in items])),
            'mrr': float(np.mean([row['mrr'] for row in items])),
        }
        for rank in ranks:
            output[f'recall_at_{rank}'] = float(np.mean([row[f'recall_at_{rank}'] for row in items]))
        summaries.append(output)
    return summaries


def bm25_tokens(text: str) -> list[str]:
    return tokenize_bm25(text)
