from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


BENCH_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def require_cache_on_d(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Run set_up.sh first; missing {name}")
    path = Path(value).resolve()
    if path.drive.lower() != "d:":
        raise RuntimeError(f"{name} must be on drive D, got: {path}")
    return path
