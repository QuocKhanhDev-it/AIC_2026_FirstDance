from __future__ import annotations

import re
import unicodedata
from typing import Sequence


_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Retrieval-oriented normalization without removing Vietnamese accents."""
    text = unicodedata.normalize("NFC", text or "").lower()
    text = text.replace("đ", "đ")
    text = _PUNCTUATION_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, *, unit: str, normalize: bool = False) -> float:
    if normalize:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if unit == "char":
        ref_units, hyp_units = list(reference), list(hypothesis)
    elif unit == "word":
        ref_units, hyp_units = reference.split(), hypothesis.split()
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    if not ref_units:
        return 0.0 if not hyp_units else 1.0
    return _levenshtein(ref_units, hyp_units) / len(ref_units)


def text_metric_record(reference: str, hypothesis: str) -> dict[str, float | bool]:
    ref_norm, hyp_norm = normalize_text(reference), normalize_text(hypothesis)
    return {
        "cer_strict": error_rate(reference, hypothesis, unit="char"),
        "wer_strict": error_rate(reference, hypothesis, unit="word"),
        "cer_norm": error_rate(ref_norm, hyp_norm, unit="char"),
        "wer_norm": error_rate(ref_norm, hyp_norm, unit="word"),
        "exact_strict": reference == hypothesis,
        "exact_norm": ref_norm == hyp_norm,
    }
