from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .config import BENCH_ROOT, load_yaml


def _winner_ocr(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if frame.empty:
        return None, {"reason": "no successful OCR results"}
    static = frame[frame.text_type == "static_text"].copy()
    eligible = set(static.loc[static.character_accuracy >= config["ocr_static_accuracy_threshold"], "model_id"])
    if not eligible:
        return None, {"reason": "no OCR model meets static character accuracy threshold"}
    combined = frame[frame.model_id.isin(eligible)].groupby("model_id", as_index=False).agg(
        cer_norm=("cer_norm", "mean"), latency=("latency_median_sec", "mean"),
        vram=("vram_peak_gb", "max"),
    ).sort_values(["cer_norm", "latency", "vram"])
    winner = str(combined.iloc[0].model_id)
    ticker = frame[(frame.model_id == winner) & (frame.text_type == "ticker_scrolling")]
    ticker_samples = int(ticker.samples.sum()) if not ticker.empty else 0
    ticker_status = "inconclusive" if ticker_samples < config["ocr_ticker_minimum_for_conclusion"] else "measured"
    return winner, {"ticker_status": ticker_status, "ticker_samples": ticker_samples}


def _winner_asr(frame: pd.DataFrame) -> tuple[str | None, dict[str, Any]]:
    if frame.empty:
        return None, {"reason": "no successful ASR results"}
    eligible = frame[frame.timestamp_valid_rate >= 1.0].sort_values(["wer_norm", "rtf_median", "vram_peak_gb"])
    if eligible.empty:
        return None, {"reason": "no ASR model has 100% valid timestamps"}
    return str(eligible.iloc[0].model_id), {}


def _plots(ocr: pd.DataFrame, asr: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not ocr.empty:
        pivot = ocr.pivot(index="model_id", columns="text_type", values="character_accuracy")
        pivot.plot(kind="bar", figsize=(11, 6), ylim=(0, 1), ylabel="Character accuracy (1-CER)")
        plt.tight_layout(); plt.savefig(plot_dir / "ocr_accuracy_by_type.png", dpi=160); plt.close()
        model = ocr.groupby("model_id", as_index=False).agg(
            character_accuracy=("character_accuracy", "mean"), latency=("latency_median_sec", "mean")
        )
        plt.figure(figsize=(8, 5)); plt.scatter(model.latency, model.character_accuracy)
        for row in model.itertuples(): plt.annotate(row.model_id, (row.latency, row.character_accuracy))
        plt.xlabel("Median latency/image (s)"); plt.ylabel("Character accuracy"); plt.tight_layout()
        plt.savefig(plot_dir / "ocr_quality_latency.png", dpi=160); plt.close()
    if not asr.empty:
        asr.set_index("model_id")[["wer_norm", "cer_norm"]].plot(kind="bar", figsize=(9, 5), ylabel="Error rate")
        plt.tight_layout(); plt.savefig(plot_dir / "asr_error_rates.png", dpi=160); plt.close()
        plt.figure(figsize=(8, 5)); plt.scatter(asr.rtf_median, asr.wer_norm)
        for row in asr.itertuples(): plt.annotate(row.model_id, (row.rtf_median, row.wer_norm))
        plt.xlabel("Median real-time factor"); plt.ylabel("WER"); plt.tight_layout()
        plt.savefig(plot_dir / "asr_wer_rtf.png", dpi=160); plt.close()


def generate_report(output_dir: Path = BENCH_ROOT / "results") -> Path:
    config = load_yaml(BENCH_ROOT / "configs/benchmark.yaml")
    ocr_path, asr_path = output_dir / "ocr_summary.csv", output_dir / "asr_summary.csv"
    ocr = pd.read_csv(ocr_path) if ocr_path.exists() and ocr_path.stat().st_size else pd.DataFrame()
    asr = pd.read_csv(asr_path) if asr_path.exists() and asr_path.stat().st_size else pd.DataFrame()
    ocr_winner, ocr_notes = _winner_ocr(ocr, config)
    asr_winner, asr_notes = _winner_asr(asr)
    _plots(ocr, asr, output_dir)

    def table(frame: pd.DataFrame) -> str:
        return "_No successful results._" if frame.empty else frame.round(4).to_markdown(index=False)

    conclusion = {
        "conclusion": {
            "ocr": {"selected_model": ocr_winner, **ocr_notes},
            "asr": {"selected_model": asr_winner, **asr_notes},
            "selection_policy": "accuracy_first; latency then VRAM for ties",
        }
    }
    report = f"""# OCR & ASR Benchmark Summary — L29

Generated with Python {sys.version.split()[0]} on {platform.platform()}.

## OCR

{table(ocr)}

## ASR

{table(asr)}

## Machine-readable conclusion

```yaml
{yaml.safe_dump(conclusion, allow_unicode=True, sort_keys=False).strip()}
```
"""
    path = output_dir / "summary_report.md"
    path.write_text(report, encoding="utf-8")
    (output_dir / "conclusion.json").write_text(json.dumps(conclusion, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(generate_report())
