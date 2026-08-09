from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


CACHE_VARS = (
    "AIC_AI_CACHE_ROOT", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE",
    "HF_ASSETS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
    "EASYOCR_MODULE_PATH", "PADDLE_PDX_CACHE_HOME", "AIC_VIETOCR_CACHE",
    "AIC_WHISPER_CACHE",
)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    expected_venv = (project_root / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_venv:
        raise RuntimeError(f"Wrong Python: {sys.prefix}; expected {expected_venv}")

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    for name in CACHE_VARS:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Missing cache variable: {name}")
        path = Path(value).resolve()
        if path.drive.lower() != "d:":
            raise RuntimeError(f"{name} is not on drive D: {path}")
        path.mkdir(parents=True, exist_ok=True)
        print('{}={}'.format(name, path))
        if name != CACHE_VARS[-1]:
            continue
        import subprocess
        modules = ('torch', 'transformers', 'paddle', 'paddleocr', 'easyocr', 'vietocr', 'jiwer')
        for module_name in modules:
            subprocess.run([sys.executable, '-c', 'import {}'.format(module_name)], check=True)
            print('isolated import {}: OK'.format(module_name))
        subprocess.run([
            sys.executable, '-c', 'import torch; assert isinstance(torch.cuda.is_available(), bool)'
        ], check=True)
        subprocess.run([
            sys.executable, '-c', 'import paddle; assert isinstance(paddle.is_compiled_with_cuda(), bool)'
        ], check=True)
        print('Environment verification passed with isolated framework imports.')
        return 0
        print(f"{name}={path}")

    for name in ("torch", "transformers", "paddle", "paddleocr", "easyocr", "vietocr", "jiwer"):
        module = importlib.import_module(name)
        print(f"import {name}: {getattr(module, '__version__', 'OK')}")

    import torch
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    import paddle
    print(f"Paddle CUDA build: {paddle.is_compiled_with_cuda()}")
    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
