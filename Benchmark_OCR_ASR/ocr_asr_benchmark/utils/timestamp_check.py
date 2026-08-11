from __future__ import annotations

from collections.abc import Iterable


def validate_timestamps(chunks: Iterable[dict], duration_sec: float) -> dict[str, float | bool | int]:
    chunks = list(chunks)
    previous_end = 0.0
    valid = True
    covered = 0.0
    for chunk in chunks:
        timestamp = chunk.get("timestamp") or chunk.get("timestamps")
        if not timestamp or len(timestamp) != 2:
            valid = False
            continue
        start, end = timestamp
        if start is None or end is None:
            valid = False
            continue
        start, end = float(start), float(end)
        if start < 0 or end <= start or end > duration_sec + 0.5 or start < previous_end - 0.05:
            valid = False
        covered += max(0.0, min(end, duration_sec) - max(start, 0.0))
        previous_end = max(previous_end, end)
    return {
        'timestamp_repaired_chunks': sum(bool(chunk.get('timestamp_repaired')) for chunk in chunks),
        "timestamp_valid": valid and bool(chunks),
        "timestamp_chunks": len(chunks),
        "timestamp_coverage": min(1.0, covered / duration_sec) if duration_sec else 0.0,
    }
