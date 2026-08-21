"""
test_pipeline.py — Unit tests cho pipeline_OCR_ASR
"""

import pandas as pd
import pytest
from pathlib import Path
from pipeline_OCR_ASR.gop_ocr_asr import gop_ocr_va_asr
from pipeline_OCR_ASR.loc_cung_token_hiem import phat_hien_token_hiem, LocCungTokenHiem
from pipeline_OCR_ASR.ocr_processor import KeyframeOCRProcessor


def test_phat_hien_token_hiem_dung():
    # 1. Các token hiếm hợp lệ
    query = "biển số 79A-12345 tại HTV và chợ Bến Thành"
    tokens = phat_hien_token_hiem(query)
    assert any("79A-12345" in t for t in tokens)
    assert any("HTV" in t for t in tokens)
    assert any("Bến" in t for t in tokens)


def test_khong_nham_tu_thuong_tieng_viet():
    # 2. Đảm bảo từ tiếng Việt thông thường (dù dài >= 6 ký tự) KHÔNG bị coi là token hiếm
    query = "nghiên cứu truyền thống trường học phương tiện giao thông"
    tokens = phat_hien_token_hiem(query)
    # Không có số, không có viết hoa, không có gạch nối -> trả về rỗng
    assert len(tokens) == 0


def test_gop_ocr_va_asr(tmp_path):
    df_master = pd.DataFrame([
        {"row_id": 0, "video_id": "V001", "kf_n": 1},
        {"row_id": 1, "video_id": "V001", "kf_n": 2},
    ])
    df_ocr = pd.DataFrame([
        {"row_id": 0, "ocr_text": "chợ Bến Thành"},
        {"row_id": 1, "ocr_text": ""},
    ])
    df_asr = pd.DataFrame([
        {"row_id": 0, "asr_text": "xin chào các bạn"},
        {"row_id": 1, "asr_text": "hôm nay là thứ hai"},
    ])

    out_file = tmp_path / "ocr_asr.parquet"
    res = gop_ocr_va_asr(df_master, df_ocr, df_asr, output_path=out_file)

    assert len(res) == 2
    assert "chợ Bến Thành . xin chào các bạn" in res.iloc[0]["text"]
    assert res.iloc[1]["text"] == "hôm nay là thứ hai"


def test_loc_cung_token_hiem():
    df_master = pd.DataFrame([
        {"row_id": 0, "video_id": "V001", "kf_n": 1, "frame_idx": 0, "pts_time": 0.0, "fps": 30.0, "title": "Test 1"},
        {"row_id": 1, "video_id": "V001", "kf_n": 2, "frame_idx": 30, "pts_time": 1.0, "fps": 30.0, "title": "Test 2"},
    ])
    df_ocr_asr = pd.DataFrame([
        {"row_id": 0, "text": "Xe ô tô mang biển số 79A-12345 chạy trên đường"},
        {"row_id": 1, "text": "Phong cảnh thiên nhiên đẹp mắt"},
    ])

    loc = LocCungTokenHiem(df_ocr_asr, df_master)
    cands = loc.tim_kiem_loc_cung("biển số 79A-12345")

    assert len(cands) == 1
    assert cands[0].row_id == 0
    assert cands[0].score >= 100.0

