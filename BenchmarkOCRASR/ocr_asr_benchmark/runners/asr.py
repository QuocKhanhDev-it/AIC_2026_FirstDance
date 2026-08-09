from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from ..config import BENCH_ROOT, load_yaml
from ..asr_loader import load_asr_adapter
from ..utils.aggregate import mean_ci, percentile
from ..utils.io import write_csv, write_json
from ..utils.resources import cleanup_models, measurement
from ..utils.text_metrics import text_metric_record
from ..utils.timestamp_check import validate_timestamps


def _summary(rows: list[dict[str, Any]], iterations: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "OK":
            groups.setdefault(row["model_id"], []).append(row)
    summaries = []
    for model_id, items in sorted(groups.items()):
        wer = mean_ci((row["wer_norm"] for row in items), iterations=iterations, seed=seed)
        summaries.append({
            "model_id": model_id, "clips": len(items), "wer_norm": wer["mean"],
            "wer_ci_low": wer["ci_low"], "wer_ci_high": wer["ci_high"],
            "cer_norm": sum(row["cer_norm"] for row in items) / len(items),
            "timestamp_valid_rate": sum(bool(row["timestamp_valid"]) for row in items) / len(items),
            "timestamp_coverage": sum(row["timestamp_coverage"] for row in items) / len(items),
            "rtf_median": percentile((row["rtf"] for row in items), 50),
            "rtf_p95": percentile((row["rtf"] for row in items), 95),
            "vram_peak_gb": max(row["vram_peak_gb"] for row in items),
        })
    return summaries


def run(args: argparse.Namespace) -> int:
    models = load_yaml(args.models)
    benchmark = load_yaml(args.benchmark)
    labels_doc = __import__("json").loads(args.labels.read_text(encoding="utf-8"))
    labels = labels_doc["clips"] if isinstance(labels_doc, dict) else labels_doc
    if args.include_unreviewed and args.limit is not None:
        labels = labels[:args.limit]
    if not args.include_unreviewed:
        if args.limit is not None:
            labels = labels[:args.limit]
        labels = [row for row in labels if row.get("review_status") == "approved"]
    if not labels:
        raise RuntimeError("No eligible ASR labels. Approve labels or use --include-unreviewed.")

    output_rows: list[dict[str, Any]] = []
    for model_cfg in models["asr"]:
        if not model_cfg.get("enabled", True):
            continue
        model_id = model_cfg["id"]
        adapter = None
        try:
            with measurement() as load_measure:
                adapter = load_asr_adapter(model_cfg, models["asr_generation"])
            adapter.predict((BENCH_ROOT / labels[0]["audio_path"]).resolve())
            for clip in labels:
                audio_path = (BENCH_ROOT / clip["audio_path"]).resolve()
                reference = " ".join(segment["text_raw"] for segment in clip["segments"]).strip()
                with measurement() as infer_measure:
                    prediction = adapter.predict(audio_path)
                hypothesis = str(prediction.get("text", "")).strip()
                chunks = prediction.get("chunks", [])
                row = {
                    "status": "OK", "model_id": model_id, "clip_id": clip["clip_id"],
                    "video_id": clip["video_id"], "audio_tags": ",".join(clip.get("audio_tags", [])),
                    "reference": reference, "prediction": hypothesis,
                    "load_time_sec": load_measure.elapsed_sec,
                    "latency_sec": infer_measure.elapsed_sec,
                    "rtf": infer_measure.elapsed_sec / float(clip["duration_sec"]),
                    "vram_peak_gb": infer_measure.vram_peak_gb,
                    "ram_delta_gb": infer_measure.rss_delta_gb,
                }
                row.update(text_metric_record(reference, hypothesis))
                row.update(validate_timestamps(chunks, float(clip["duration_sec"])))
                output_rows.append(row)
        except RuntimeError as exc:
            status = "SKIPPED_OOM" if "out of memory" in str(exc).lower() else "FAILED"
            output_rows.append({"status": status, "model_id": model_id, "error": str(exc), "traceback": traceback.format_exc()})
        except Exception as exc:
            output_rows.append({"status": "FAILED", "model_id": model_id, "error": str(exc), "traceback": traceback.format_exc()})
        finally:
            cleanup_models(adapter)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "asr_results.csv", output_rows)
    summary = _summary(output_rows, benchmark["bootstrap_iterations"], benchmark["seed"])
    write_csv(args.output_dir / "asr_summary.csv", summary)
    write_json(args.output_dir / "asr_summary.json", summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ASR benchmark")
    parser.add_argument("--models", type=Path, default=BENCH_ROOT / "configs/models.yaml")
    parser.add_argument("--benchmark", type=Path, default=BENCH_ROOT / "configs/benchmark.yaml")
    parser.add_argument("--labels", type=Path, default=BENCH_ROOT / "eval_data/asr/transcripts.json")
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / "results")
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument('--limit', type=int)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
