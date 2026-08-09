from __future__ import annotations

from pathlib import Path
from typing import Any

from .model_loader import TransformersWhisperAdapter


class TimestampSafeWhisperAdapter(TransformersWhisperAdapter):
    def predict(self, audio_path: Path) -> dict[str, Any]:
        result = super().predict(audio_path)
        import soundfile as sf
        duration = float(sf.info(audio_path).duration)
        previous_end = 0.0
        for chunk in result.get('chunks', []):
            timestamp = chunk.get('timestamp') or chunk.get('timestamps') or (None, None)
            start, end = timestamp
            repaired = start is None or end is None
            start = previous_end if start is None else max(0.0, float(start))
            end = duration if end is None else min(duration, float(end))
            chunk['timestamp'] = (start, max(start, end))
            chunk['timestamp_repaired'] = repaired
            previous_end = max(previous_end, end)
        return result


def load_asr_adapter(model_config: dict[str, Any], generation: dict[str, Any]) -> Any:
    if model_config['family'] == 'transformers_whisper':
        return TimestampSafeWhisperAdapter(model_config, generation)
    raise ValueError('Unknown ASR family: {}'.format(model_config['family']))
