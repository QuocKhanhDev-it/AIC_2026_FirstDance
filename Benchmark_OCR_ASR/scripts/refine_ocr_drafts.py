from __future__ import annotations

import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from ocr_asr_benchmark.utils.io import load_jsonl, write_jsonl


# Only visually unambiguous crops are corrected here. They remain unapproved
# until a human reviewer confirms them against the contact sheets.
CORRECTIONS = {
    'ocr_L29_V001_01': 'CÀ MAU',
    'ocr_L29_V001_02': 'PHÁT',
    'ocr_L29_V001_05': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V001_06': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V001_07': 'ĐÔI MẮT',
    'ocr_L29_V001_08': 'MÊ KÔNG',
    'ocr_L29_V001_10': 'ĐÔI MẮT',
    'ocr_L29_V001_11': 'ĐÔI MẮT',
    'ocr_L29_V001_12': 'ĐÀI TRUYỀN HÌNH TP. HỒ CHÍ MINH',
    'ocr_L29_V004_01': 'CÀ MAU',
    'ocr_L29_V004_04': 'Gác kèo',
    'ocr_L29_V004_05': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V004_07': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V004_08': 'MẮT',
    'ocr_L29_V004_09': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V004_10': 'MẮT',
    'ocr_L29_V008_01': 'ĐÔI MẮT',
    'ocr_L29_V008_03': 'Làng tôm khô Rạch Gốc',
    'ocr_L29_V008_06': 'HOÀNG',
    'ocr_L29_V008_07': 'Làng tôm khô Rạch Gốc',
    'ocr_L29_V008_08': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V008_10': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V008_11': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V008_12': 'ĐÀI TRUYỀN HÌNH TP. HỒ CHÍ MINH',
    'ocr_L29_V012_01': 'U MINH HA NATIONAL PARK',
    'ocr_L29_V012_04': 'Dẫn chương trình',
    'ocr_L29_V012_05': 'cuối lá dừa',
    'ocr_L29_V012_06': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V012_08': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V012_09': 'ĐÔI MẮT',
    'ocr_L29_V012_10': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V012_11': 'ĐÔI MẮT',
    'ocr_L29_V012_12': 'TMT Group',
    'ocr_L29_V014_01': 'ĐÔI MẮT',
    'ocr_L29_V014_03': 'dẫn dắt Nguyễn Phích',
    'ocr_L29_V014_04': 'MẮT',
    'ocr_L29_V014_05': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V014_06': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V014_07': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V014_08': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V014_09': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V014_10': 'ĐÔI MẮT',
    'ocr_L29_V014_11': 'ĐÔI MẮT MÊ KÔNG',
    'ocr_L29_V014_12': 'Hoài Nam',
}


def main() -> int:
    path = BENCH_ROOT / 'eval_data/ocr/ocr_labels.jsonl'
    rows = load_jsonl(path)
    negatives = corrected = 0
    for row in rows:
        if not str(row.get('text_raw', '')).strip():
            row['text_type'] = 'no_text'
            row['legibility'] = 'no_text'
            row['notes'] = 'Intentional negative sample: no visible target text'
            negatives += 1
        elif row['sample_id'] in CORRECTIONS:
            row['text_raw'] = CORRECTIONS[row['sample_id']]
            row['legibility'] = 'visual_draft'
            row['notes'] = 'Visually corrected draft; human confirmation still required'
            corrected += 1
        row['review_status'] = 'needs_review'
    write_jsonl(path, rows)
    print(f'Refined {corrected} clear OCR drafts; marked {negatives} negative samples')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
