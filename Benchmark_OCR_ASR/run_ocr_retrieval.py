from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from ocr_asr_benchmark.config import BENCH_ROOT, load_yaml
from ocr_asr_benchmark.retrieval_v2 import (
    bm25_tokens, first_relevant_rank, query_metrics,
    reciprocal_rank_fusion, summarize_queries,
)
from ocr_asr_benchmark.utils.io import load_jsonl, write_csv, write_json


def _frame_map(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _documents(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config['dataset_root'])
    documents = []
    for domain, domain_cfg in config['sample_policy']['domains'].items():
        videos = domain_cfg['dev_videos'] + domain_cfg['holdout_videos']
        for video_id in videos:
            metadata = json.loads((root / 'media-info' / f'{video_id}.json').read_text(encoding='utf-8'))
            metadata_text = f'{metadata.get("title", "")} {metadata.get("description", "")}'
            for mapped in _frame_map(root / 'map-keyframes' / f'{video_id}.csv'):
                documents.append({
                    'domain': domain, 'video_id': video_id,
                    'kf_n': int(mapped['n']), 'frame_idx': int(mapped['frame_idx']),
                    'metadata_text': metadata_text, 'ocr_text': '',
                })
    return sorted(documents, key=lambda row: (row['video_id'], row['frame_idx']))


def _accepted_ocr(gate_dir: Path) -> tuple[dict[tuple[str, int], str], dict[str, Any]]:
    summary = json.loads((gate_dir / 'gate_summary.json').read_text(encoding='utf-8'))
    threshold = summary.get('selected_threshold')
    if threshold is None:
        return {}, summary
    accepted: dict[tuple[str, int], list[str]] = {}
    with (gate_dir / 'gate_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            texts = [
                item['text'] for item in json.loads(row['candidates'])
                if item.get('text') and float(item.get('confidence') or 0.0) >= float(threshold)
            ]
            if texts:
                accepted[(row['video_id'], int(row['frame_idx']))] = texts
    return {key: ' '.join(value) for key, value in accepted.items()}, summary


def _bm25(corpus: list[str], config: dict[str, Any]) -> BM25Okapi:
    tokens = [bm25_tokens(text) or ['__empty__'] for text in corpus]
    return BM25Okapi(tokens, k1=float(config['retrieval']['bm25_k1']), b=float(config['retrieval']['bm25_b']))


def _evaluate(
    documents: list[dict[str, Any]], queries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_bm25 = _bm25([row['metadata_text'] for row in documents], config)
    ocr_bm25 = _bm25([row['ocr_text'] for row in documents], config)
    identity = {(row['video_id'], row['frame_idx']): index for index, row in enumerate(documents)}
    ranks = config['retrieval']['ranks']
    rows = []
    for query in queries:
        query_tokens = bm25_tokens(query['query'])
        metadata_scores = metadata_bm25.get_scores(query_tokens)
        ocr_scores = ocr_bm25.get_scores(query_tokens)
        fused_scores = reciprocal_rank_fusion(
            metadata_scores, ocr_scores, k=int(config['retrieval']['rrf_k']),
        )
        relevant = set()
        for target in query['relevant_targets']:
            if target.get('frame_idx') is None:
                relevant.update(
                    index for index, document in enumerate(documents)
                    if document['video_id'] == target['video_id']
                )
            else:
                key = (target['video_id'], int(target['frame_idx']))
                if key in identity:
                    relevant.add(identity[key])
        if not relevant:
            raise ValueError(f'Query target is outside the evaluation corpus: {query["query_id"]}')
        systems = {
            'metadata': metadata_scores,
            'ocr': ocr_scores,
            'metadata_plus_ocr': fused_scores,
        }
        for system, scores in systems.items():
            order = np.argsort(-np.asarray(scores), kind='stable')
            first_rank = first_relevant_rank(order, relevant)
            row = {
                'query_id': query['query_id'], 'domain': query['domain'],
                'query_type': query['query_type'], 'query': query['query'],
                'system': system,
            }
            row.update(query_metrics(first_rank, ranks))
            rows.append(row)
    return rows, summarize_queries(rows, ranks)


def _decision(summary: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    lookup = {(row['system'], row['domain'], row['query_type']): row for row in summary}
    metadata = lookup[('metadata', 'ALL', 'ALL')]['final_score']
    combined = lookup[('metadata_plus_ocr', 'ALL', 'ALL')]['final_score']
    metadata_control = lookup[('metadata', 'ALL', 'metadata_control')]['final_score']
    combined_control = lookup[('metadata_plus_ocr', 'ALL', 'metadata_control')]['final_score']
    retrieval_passed = combined > metadata and combined_control >= metadata_control
    gate_passed = bool(gate.get('gate_passed'))
    if not gate_passed:
        status = 'drop_ocr_channel_false_positive'
    elif not retrieval_passed:
        status = 'drop_ocr_channel_no_retrieval_gain'
    else:
        status = 'selected_for_full_run'
    return {
        'status': status, 'gate_passed': gate_passed,
        'retrieval_passed': retrieval_passed,
        'metadata_final_score': metadata,
        'metadata_plus_ocr_final_score': combined,
        'metadata_control_final_score': metadata_control,
        'metadata_plus_ocr_control_final_score': combined_control,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='BM25 metadata vs metadata+OCR ablation')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--config', type=Path, default=BENCH_ROOT / 'configs/ocr_v2.yaml')
    parser.add_argument('--queries', type=Path, default=BENCH_ROOT / 'eval_data/ocr_v2/retrieval_queries.jsonl')
    parser.add_argument('--gate-root', type=Path, default=BENCH_ROOT / 'results/v2/gate')
    parser.add_argument('--output-dir', type=Path, default=BENCH_ROOT / 'results/v2/retrieval')
    args = parser.parse_args()
    config = load_yaml(args.config)
    queries = load_jsonl(args.queries)
    if len(queries) != config['retrieval']['queries_total'] or any(row.get('review_status') != 'approved' for row in queries):
        raise RuntimeError('Exactly 40 approved retrieval queries are required')
    documents = _documents(config)
    ocr, gate = _accepted_ocr(args.gate_root / args.model_id)
    for document in documents:
        document['ocr_text'] = ocr.get((document['video_id'], document['frame_idx']), '')
    rows, summary = _evaluate(documents, queries, config)
    decision = _decision(summary, gate)
    output = args.output_dir / args.model_id
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'retrieval_results.csv', rows)
    write_csv(output / 'retrieval_summary.csv', summary)
    write_json(output / 'retrieval_conclusion.json', decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
