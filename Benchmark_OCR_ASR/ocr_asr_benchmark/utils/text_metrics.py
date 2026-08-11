from __future__ import annotations

import re
import unicodedata
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence


_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Retrieval-oriented normalization without removing Vietnamese accents."""
    text = unicodedata.normalize("NFC", text or "").casefold()
    text = _PUNCTUATION_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize_bm25(text: str) -> list[str]:
    """The single tokenizer shared by OCR metrics and the BM25 evaluation."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


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


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> float:
    """Corpus WER: total word edits divided by total reference words."""
    edits = 0
    reference_words = 0
    false_positive_documents = 0
    for reference, hypothesis in pairs:
        ref_tokens = tokenize_bm25(reference)
        hyp_tokens = tokenize_bm25(hypothesis)
        if not ref_tokens:
            false_positive_documents += bool(hyp_tokens)
            continue
        edits += _levenshtein(ref_tokens, hyp_tokens)
        reference_words += len(ref_tokens)
    if reference_words:
        return edits / reference_words
    return 1.0 if false_positive_documents else 0.0


def token_prf(
    reference: str,
    hypothesis: str,
    *,
    idf: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Bag-of-words precision/recall/F1, optionally weighted by BM25 IDF."""
    ref = Counter(tokenize_bm25(reference))
    hyp = Counter(tokenize_bm25(hypothesis))

    def weight(token: str) -> float:
        return float(idf.get(token, 1.0)) if idf is not None else 1.0

    overlap = sum(min(count, hyp[token]) * weight(token) for token, count in ref.items())
    ref_total = sum(count * weight(token) for token, count in ref.items())
    hyp_total = sum(count * weight(token) for token, count in hyp.items())
    precision = overlap / hyp_total if hyp_total else (1.0 if not ref_total else 0.0)
    recall = overlap / ref_total if ref_total else (1.0 if not hyp_total else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"token_precision": precision, "token_recall": recall, "token_f1": f1}


def bm25_idf(texts: Iterable[str]) -> dict[str, float]:
    documents = [set(tokenize_bm25(text)) for text in texts]
    document_count = len(documents)
    document_frequency = Counter(token for document in documents for token in document)
    return {
        token: math.log(1.0 + (document_count - count + 0.5) / (count + 0.5))
        for token, count in document_frequency.items()
    }
