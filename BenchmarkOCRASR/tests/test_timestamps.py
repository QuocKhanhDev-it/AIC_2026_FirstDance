from ocr_asr_benchmark.utils.timestamp_check import validate_timestamps


def test_valid_timestamps() -> None:
    result = validate_timestamps([
        {'timestamp': (0.0, 1.0)}, {'timestamp': (1.0, 2.5)},
    ], 3.0)
    assert result['timestamp_valid'] is True
    assert result['timestamp_chunks'] == 2


def test_overlapping_timestamps_are_invalid() -> None:
    result = validate_timestamps([
        {'timestamp': (0.0, 2.0)}, {'timestamp': (1.0, 3.0)},
    ], 3.0)
    assert result['timestamp_valid'] is False
