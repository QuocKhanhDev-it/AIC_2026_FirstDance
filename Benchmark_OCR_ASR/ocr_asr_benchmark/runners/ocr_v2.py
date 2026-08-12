from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import BENCH_ROOT, load_yaml
from ..ocr_v2_adapters import EasyOCRDetector, load_recognizer
from ..ocr_v2_eval import box_in_roi, choose_threshold, retrieval_tokens, summarize_recognizer, threshold_metrics
from ..utils.geometry import bbox_iou
from ..utils.io import load_jsonl, write_csv, write_json, write_jsonl
from ..utils.resources import cleanup_models, measurement
from ..utils.text_metrics import bm25_idf, text_metric_record


def _image_path(config: dict[str, Any], row: dict[str, Any]) -> Path:
    path = Path(row['image_path'])
    return path if path.is_absolute() else Path(config['dataset_root']) / path


def _crop(image: Image.Image, bbox: list[float], padding_ratio: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = map(float, bbox)
    padding_x = (x2 - x1) * padding_ratio
    padding_y = (y2 - y1) * padding_ratio
    return image.crop((
        max(0, x1 - padding_x), max(0, y1 - padding_y),
        min(width, x2 + padding_x), min(height, y2 + padding_y),
    )).convert('RGB')


def _model_config(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = [item for item in config['recognizers'] if item['id'] == model_id and item.get('enabled', True)]
    if len(models) != 1:
        raise ValueError(f'Expected one enabled recognizer named {model_id}, got {len(models)}')
    return models[0]


def run_recognizer(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    labels = load_jsonl(args.positive_labels)
    if not args.include_unreviewed:
        labels = [
            row for row in labels
            if row.get('review_status') == 'approved' and row.get('second_review_status') == 'approved'
        ]
    if args.limit is not None:
        labels = labels[:args.limit]
    if not labels:
        raise RuntimeError('No approved OCR v2 positive labels')
    model_cfg = _model_config(config, args.model_id)
    idf = bm25_idf(sample['text_raw'] for sample in labels)
    model = None
    rows: list[dict[str, Any]] = []
    try:
        model = load_recognizer(model_cfg)
        for sample in labels:
            image_path = _image_path(config, sample)
            with Image.open(image_path) as image:
                crop = _crop(image.convert('RGB'), sample['bbox_xyxy'], config['crop_padding_ratio'])
            with measurement() as resource:
                result = model.recognize_line(crop)
            metrics = text_metric_record(sample['text_raw'], result.text)
            row = {
                'status': 'OK', 'model_id': args.model_id,
                'sample_id': sample['sample_id'], 'domain': sample['domain'],
                'split': sample['split'], 'video_id': sample['video_id'],
                'frame_idx': sample['frame_idx'], 'semantic_type': sample['semantic_type'],
                'reference': sample['text_raw'], 'prediction': result.text,
                'confidence': result.confidence, 'latency_sec': resource.elapsed_sec,
                'vram_peak_gb': resource.vram_peak_gb, 'ram_delta_gb': resource.rss_delta_gb,
            }
            row.update(metrics)
            row.update(retrieval_tokens(sample['text_raw'], result.text, idf=idf))
            rows.append(row)
    except Exception as exc:
        rows.append({
            'status': 'FAILED', 'model_id': args.model_id,
            'error': str(exc), 'traceback': traceback.format_exc(),
        })
    finally:
        cleanup_models(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'recognizer_results.csv', rows)
    write_jsonl(args.output_dir / 'recognizer_results.jsonl', rows)
    summary = summarize_recognizer(rows)
    write_csv(args.output_dir / 'recognizer_summary.csv', summary)
    write_json(args.output_dir / 'recognizer_summary.json', summary)
    return 0 if any(row.get('status') == 'OK' for row in rows) else 1


def _gate_sample(
    config: dict[str, Any], roi: dict[str, Any], detector: Any,
    recognizer: Any, sample: dict[str, Any],
) -> dict[str, Any]:
    image_path = _image_path(config, sample)
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    width, height = image.size
    profile = roi['profiles'][sample['domain']]
    candidates = []
    with measurement() as resource:
        detections = detector.detect(image)
        for detection in detections:
            bbox = detection['bbox_xyxy']
            if not box_in_roi(bbox, width, height, profile['include'], profile.get('exclude', [])):
                continue
            result = recognizer.recognize_line(_crop(image, bbox, config['crop_padding_ratio']))
            candidates.append({
                'bbox_xyxy': bbox,
                'text': result.text,
                'confidence': result.confidence,
                'matched_target': bool(sample.get('bbox_xyxy'))
                and bbox_iou(bbox, sample['bbox_xyxy']) >= 0.5,
            })
    return {
        'status': 'OK', 'sample_id': sample['sample_id'],
        'domain': sample['domain'], 'split': sample['split'],
        'video_id': sample['video_id'], 'frame_idx': sample['frame_idx'],
        'target_present': sample['label_kind'] == 'positive',
        'candidates': candidates, 'latency_sec': resource.elapsed_sec,
        'vram_peak_gb': resource.vram_peak_gb, 'ram_delta_gb': resource.rss_delta_gb,
    }


def run_gate(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    roi = load_yaml(args.roi)
    for domain in ('L21', 'L29'):
        profile = roi.get('profiles', {}).get(domain, {})
        if profile.get('review_status') != 'approved' or not profile.get('include'):
            raise RuntimeError(f'ROI profile {domain} must be reviewed and approved')
    positives = load_jsonl(args.positive_labels)
    negatives = load_jsonl(args.negative_labels)
    labels = positives + negatives
    if not args.include_unreviewed:
        labels = [
            row for row in labels
            if row.get('review_status') == 'approved' and row.get('second_review_status') == 'approved'
        ]
    if args.limit is not None:
        labels = labels[:args.limit]
    if not labels:
        raise RuntimeError('No approved OCR v2 gate labels')
    model_cfg = _model_config(config, args.model_id)
    recognizer = detector = None
    rows: list[dict[str, Any]] = []
    try:
        detector = EasyOCRDetector(config['fixed_detector'])
        recognizer = load_recognizer(model_cfg)
        rows = [_gate_sample(config, roi, detector, recognizer, sample) for sample in labels]
    finally:
        cleanup_models(recognizer, detector)
    dev_rows = [row for row in rows if row['split'] == 'dev']
    holdout_rows = [row for row in rows if row['split'] == 'holdout']
    threshold, curve = choose_threshold(
        dev_rows, config['confidence_thresholds'],
        min_recall=config['gate']['min_positive_recall'],
    )
    holdout = threshold_metrics(holdout_rows, threshold) if threshold is not None else None
    passed = bool(
        holdout
        and holdout['positive_recall'] >= config['gate']['min_positive_recall']
        and holdout['false_positive_rate'] <= config['gate']['max_false_positive_rate']
    )
    summary = {
        'model_id': args.model_id, 'selected_threshold': threshold,
        'dev_curve': curve, 'holdout': holdout, 'gate_passed': passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = [dict(row, candidates=json.dumps(row['candidates'], ensure_ascii=False)) for row in rows]
    write_csv(args.output_dir / 'gate_results.csv', csv_rows)
    write_json(args.output_dir / 'gate_summary.json', summary)
    write_csv(args.output_dir / 'gate_dev_curve.csv', curve)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='OCR v2 recognizer and ROI/confidence benchmark')
    parser.add_argument('mode', choices=['recognizer', 'gate'])
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--config', type=Path, default=BENCH_ROOT / 'configs/ocr_v2.yaml')
    parser.add_argument('--roi', type=Path, default=BENCH_ROOT / 'configs/roi_v2.yaml')
    parser.add_argument('--positive-labels', type=Path, default=BENCH_ROOT / 'eval_data/ocr_v2/positive_labels.jsonl')
    parser.add_argument('--negative-labels', type=Path, default=BENCH_ROOT / 'eval_data/ocr_v2/negative_labels.jsonl')
    parser.add_argument('--output-dir', type=Path, default=BENCH_ROOT / 'results/v2')
    parser.add_argument('--include-unreviewed', action='store_true')
    parser.add_argument('--limit', type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_recognizer(args) if args.mode == 'recognizer' else run_gate(args)


if __name__ == '__main__':
    raise SystemExit(main())
