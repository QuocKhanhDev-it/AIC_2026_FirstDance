from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import BENCH_ROOT, load_yaml
from ..model_loader import load_ocr_adapter
from ..utils.aggregate import mean_ci, percentile
from ..utils.geometry import bbox_iou
from ..utils.io import load_jsonl, write_csv, write_json
from ..utils.resources import cleanup_models, measurement
from ..utils.text_metrics import text_metric_record


def _match(predictions: list[dict[str, Any]], target_bbox: list[float]) -> tuple[dict[str, Any] | None, float]:
    if not predictions:
        return None, 0.0
    scored = [(bbox_iou(item["bbox_xyxy"], target_bbox), item) for item in predictions]
    score, item = max(scored, key=lambda pair: pair[0])
    return item, score


def _summary(rows: list[dict[str, Any]], iterations: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "OK":
            groups.setdefault((row["model_id"], row["text_type"]), []).append(row)
    summaries = []
    for (model_id, text_type), items in sorted(groups.items()):
        cer = mean_ci((row["cer_norm_full"] for row in items), iterations=iterations, seed=seed)
        summaries.append({
            "model_id": model_id, "text_type": text_type, "samples": len(items),
            "cer_norm": cer["mean"], "cer_ci_low": cer["ci_low"], "cer_ci_high": cer["ci_high"],
            "character_accuracy": 1.0 - cer["mean"],
            "wer_norm": sum(row["wer_norm_full"] for row in items) / len(items),
            "exact_norm": sum(bool(row["exact_norm_full"]) for row in items) / len(items),
            "detection_recall_iou50": sum(row["iou"] >= 0.5 for row in items) / len(items),
            "latency_median_sec": percentile((row["latency_full_sec"] for row in items), 50),
            "latency_p95_sec": percentile((row["latency_full_sec"] for row in items), 95),
            "vram_peak_gb": max(row["vram_peak_gb"] for row in items),
        })
        if text_type == 'no_text':
            summaries[-1]['false_positive_rate'] = sum(
                bool(str(row['prediction_full']).strip()) for row in items
            ) / len(items)
    return summaries


def run(args: argparse.Namespace) -> int:
    models = load_yaml(args.models)
    benchmark = load_yaml(args.benchmark)
    labels = load_jsonl(args.labels)
    if args.limit is not None:
        labels = labels[:args.limit]
    if not args.include_unreviewed:
        labels = [row for row in labels if row.get("review_status") == "approved"]
    if not labels:
        raise RuntimeError("No eligible OCR labels. Approve labels or use --include-unreviewed.")

    output_rows: list[dict[str, Any]] = []
    for model_cfg in models["ocr"]:
        if not model_cfg.get("enabled", True):
            continue
        model_id = model_cfg["id"]
        adapter = None
        try:
            with measurement() as load_measure:
                adapter = load_ocr_adapter(model_cfg)
            adapter.predict((BENCH_ROOT / labels[0]["image_path"]).resolve())
            for sample in labels:
                image_path = (BENCH_ROOT / sample["image_path"]).resolve()
                target_bbox = sample["bbox_xyxy"]
                with measurement() as full_measure:
                    predictions = adapter.predict(image_path)
                matched, iou = _match(predictions, target_bbox)
                full_text = matched["text"] if matched else ""
                with Image.open(image_path) as image:
                    crop = image.convert("RGB").crop(tuple(target_bbox))
                    with measurement() as crop_measure:
                        crop_text = adapter.recognize_crop(crop)
                full_metrics = text_metric_record(sample["text_raw"], full_text)
                crop_metrics = text_metric_record(sample["text_raw"], crop_text)
                row = {
                    "status": "OK", "model_id": model_id, "sample_id": sample["sample_id"],
                    "video_id": sample["video_id"], "frame_idx": sample["frame_idx"],
                    "text_type": sample["text_type"], "reference": sample["text_raw"],
                    "prediction_full": full_text, "prediction_crop": crop_text, "iou": iou,
                    "detected": matched is not None, "load_time_sec": load_measure.elapsed_sec,
                    "latency_full_sec": full_measure.elapsed_sec,
                    "latency_crop_sec": crop_measure.elapsed_sec,
                    "vram_peak_gb": max(full_measure.vram_peak_gb, crop_measure.vram_peak_gb),
                    "ram_delta_gb": max(full_measure.rss_delta_gb, crop_measure.rss_delta_gb),
                }
                row.update({f"{key}_full": value for key, value in full_metrics.items()})
                row.update({f"{key}_crop": value for key, value in crop_metrics.items()})
                output_rows.append(row)
        except RuntimeError as exc:
            status = "SKIPPED_OOM" if "out of memory" in str(exc).lower() else "FAILED"
            output_rows.append({"status": status, "model_id": model_id, "error": str(exc), "traceback": traceback.format_exc()})
        except Exception as exc:
            output_rows.append({"status": "FAILED", "model_id": model_id, "error": str(exc), "traceback": traceback.format_exc()})
        finally:
            cleanup_models(adapter)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ocr_results.csv", output_rows)
    summary = _summary(output_rows, benchmark["bootstrap_iterations"], benchmark["seed"])
    write_csv(args.output_dir / "ocr_summary.csv", summary)
    write_json(args.output_dir / "ocr_summary.json", summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OCR benchmark")
    parser.add_argument("--models", type=Path, default=BENCH_ROOT / "configs/models.yaml")
    parser.add_argument("--benchmark", type=Path, default=BENCH_ROOT / "configs/benchmark.yaml")
    parser.add_argument("--labels", type=Path, default=BENCH_ROOT / "eval_data/ocr/ocr_labels.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / "results")
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument('--limit', type=int)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
