from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import psutil


def reset_gpu_peak() -> None:
    import sys
    if 'torch' not in sys.modules:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_peak_gb() -> float:
    import sys
    if 'torch' not in sys.modules:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 3)
        except Exception:
            return 0.0
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def cleanup_models(*models: object) -> None:
    for model in models:
        del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


@dataclass
class Measurement:
    started_at: float
    rss_start: int
    elapsed_sec: float = 0.0
    rss_delta_gb: float = 0.0
    vram_peak_gb: float = 0.0

    def __enter__(self) -> "Measurement":
        reset_gpu_peak()
        self.started_at = time.perf_counter()
        self.rss_start = psutil.Process().memory_info().rss
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_sec = time.perf_counter() - self.started_at
        rss_end = psutil.Process().memory_info().rss
        self.rss_delta_gb = max(0, rss_end - self.rss_start) / (1024 ** 3)
        self.vram_peak_gb = gpu_peak_gb()


def measurement() -> Measurement:
    return Measurement(started_at=0.0, rss_start=0)
