from scripts.validate_labels import BENCH_ROOT, validate_asr, validate_ocr


def test_current_draft_labels_are_structurally_valid() -> None:
    assert validate_ocr(BENCH_ROOT / 'eval_data/ocr/ocr_labels.jsonl', False) == []
    assert validate_asr(BENCH_ROOT / 'eval_data/asr/transcripts.json', False) == []
