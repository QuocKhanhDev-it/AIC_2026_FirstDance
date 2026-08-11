from ocr_asr_benchmark.utils.text_metrics import error_rate, normalize_text, text_metric_record


def test_normalization_preserves_vietnamese_accents() -> None:
    assert normalize_text('  Cà Mau, VIỆT NAM! ') == 'cà mau việt nam'


def test_normalized_metrics_ignore_case_and_punctuation() -> None:
    result = text_metric_record('ĐÀI TRUYỀN HÌNH', 'đài truyền hình!')
    assert result['exact_norm'] is True
    assert result['cer_norm'] == 0.0


def test_empty_reference_penalizes_false_positive() -> None:
    assert error_rate('', '', unit='char') == 0.0
    assert error_rate('', 'chữ giả', unit='char') == 1.0
