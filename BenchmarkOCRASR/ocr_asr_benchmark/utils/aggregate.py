from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def mean_ci(values: Iterable[float], *, iterations: int = 2000, seed: int = 2026) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(iterations, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def percentile(values: Iterable[float], value: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, value)) if len(array) else float("nan")
