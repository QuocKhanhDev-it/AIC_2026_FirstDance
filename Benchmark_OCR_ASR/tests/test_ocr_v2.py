from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from ocr_asr_benchmark.ocr_v2_adapters import EasyOCRDetector, EasyOCRRecognizer, VietOCRRecognizer
from ocr_asr_benchmark.ocr_v2_eval import (
    box_in_roi, choose_threshold, paired_cluster_bootstrap, threshold_metrics,
)
from ocr_asr_benchmark.retrieval_v2 import query_metrics, reciprocal_rank_fusion
from ocr_asr_benchmark.report import _winner_ocr
from ocr_asr_benchmark.utils.text_metrics import bm25_idf, corpus_wer, token_prf, tokenize_bm25


def test_bm25_tokenizer_preserves_accents_and_digits() -> None:
    assert tokenize_bm25('  Hồ Chí Minh, 2026! ') == ['hồ', 'chí', 'minh', '2026']


def test_corpus_wer_weights_reference_tokens() -> None:
    pairs = [('một hai ba bốn', 'một hai ba bốn'), ('đúng', 'sai')]
    assert corpus_wer(pairs) == 0.2


def test_token_f1_is_order_insensitive_for_bm25() -> None:
    result = token_prf('cà mau việt nam', 'việt nam cà mau')
    assert result == {'token_precision': 1.0, 'token_recall': 1.0, 'token_f1': 1.0}


def test_bm25_idf_weights_rare_tokens_more() -> None:
    idf = bm25_idf(['người cà mau', 'người thành phố', 'người phóng viên'])
    assert idf['cà'] > idf['người']


def test_recognizer_only_easyocr_does_not_call_detector() -> None:
    class Reader:
        def recognize(self, *_args, **_kwargs):
            return [([0, 1, 0, 1], 'CÀ MAU', 0.9)]

        def detect(self, *_args, **_kwargs):
            raise AssertionError('detector must not be called')

    adapter = EasyOCRRecognizer.__new__(EasyOCRRecognizer)
    adapter.reader = Reader()
    result = adapter.recognize_line(Image.new('RGB', (20, 10), 'white'))
    assert result.text == 'CÀ MAU'
    assert result.confidence == 0.9


def test_fixed_easyocr_detector_does_not_call_recognizer() -> None:
    class Reader:
        def detect(self, *_args, **_kwargs):
            return [[[1, 9, 2, 8]]], [[]]

        def readtext(self, *_args, **_kwargs):
            raise AssertionError('recognizer must not be called')

    adapter = EasyOCRDetector.__new__(EasyOCRDetector)
    adapter.reader = Reader()
    output = adapter.detect(Image.new('RGB', (20, 10), 'white'))
    assert output[0]['bbox_xyxy'] == [1.0, 2.0, 9.0, 8.0]


def test_vietocr_confidence_comes_from_predictor_probability() -> None:
    class Predictor:
        def predict(self, _image, return_prob=False):
            assert return_prob is True
            return 'HỒ CHÍ MINH', np.float32(0.875)

    adapter = VietOCRRecognizer.__new__(VietOCRRecognizer)
    adapter.predictor = Predictor()
    result = adapter.recognize_line(Image.new('RGB', (20, 10), 'white'))
    assert result.text == 'HỒ CHÍ MINH'
    assert abs(result.confidence - 0.875) < 1e-6


def test_roi_uses_box_center_and_exclusion_masks() -> None:
    assert box_in_roi([10, 60, 30, 80], 100, 100, [0.0, 0.5, 1.0, 0.9], [])
    assert not box_in_roi([80, 60, 95, 80], 100, 100, [0.0, 0.5, 1.0, 0.9], [[0.75, 0.5, 1.0, 0.9]])


def test_gate_is_frame_level_with_multiple_boxes() -> None:
    rows = [
        {'target_present': True, 'candidates': [
            {'text': 'rác', 'confidence': 0.9, 'matched_target': False},
            {'text': 'đúng', 'confidence': 0.8, 'matched_target': True},
        ]},
        {'target_present': False, 'candidates': [
            {'text': 'logo', 'confidence': 0.4, 'matched_target': False},
        ]},
    ]
    result = threshold_metrics(rows, 0.8)
    assert result['positive_recall'] == 1.0
    assert result['false_positive_rate'] == 0.0
    selected, _curve = choose_threshold(rows, [0.0, 0.8, 0.95], min_recall=0.8)
    assert selected == 0.8


def test_paired_bootstrap_keeps_video_clusters() -> None:
    rows = []
    for model, predictions in {'good': ['a b', 'c d'], 'bad': ['x y', 'x y']}.items():
        for index, prediction in enumerate(predictions):
            rows.append({
                'status': 'OK', 'split': 'holdout', 'model_id': model,
                'video_id': f'V{index}', 'sample_id': f'S{index}',
                'reference': ['a b', 'c d'][index], 'prediction': prediction,
                'exact_norm': prediction == ['a b', 'c d'][index],
            })
    delta = paired_cluster_bootstrap(rows, 'good', 'bad', metric='corpus_wer_norm', iterations=100, seed=1)
    assert delta['ci_high'] < 0


def test_rrf_only_adds_ocr_rank_for_positive_ocr_scores() -> None:
    fused = reciprocal_rank_fusion([3.0, 2.0, 1.0], [0.0, 5.0, 0.0], k=60)
    assert fused[1] > fused[0]
    metrics = query_metrics(5, [1, 5, 20])
    assert metrics['final_score'] == 2 / 3


def test_legacy_report_cannot_select_from_13_samples_or_cer() -> None:
    frame = pd.DataFrame([
        {'model_id': 'viet', 'text_type': 'static_text', 'samples': 13, 'wer_norm': 0.4,
         'exact_norm': 0.38, 'latency_median_sec': 0.4, 'vram_peak_gb': 0.7},
        {'model_id': 'easy', 'text_type': 'static_text', 'samples': 13, 'wer_norm': 0.66,
         'exact_norm': 0.15, 'latency_median_sec': 0.2, 'vram_peak_gb': 0.6},
    ])
    winner, notes = _winner_ocr(frame, {'ocr_static_minimum_for_selection': 50})
    assert winner is None
    assert 'insufficient static samples' in notes['reason']
    frame['samples'] = 50
    winner, _notes = _winner_ocr(frame, {'ocr_static_minimum_for_selection': 50, 'ocr_ticker_minimum_for_conclusion': 10})
    assert winner == 'viet'
