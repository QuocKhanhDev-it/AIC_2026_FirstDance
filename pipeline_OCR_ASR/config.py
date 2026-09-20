"""
config.py — Cấu hình hệ thống OCR & ASR Production (Giai đoạn 1)
"""

from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu chỉ mục & output
INDEX_DIR = BASE_DIR / "index"
OUTPUT_DIR = BASE_DIR / "pipeline_OCR_ASR" / "output"

# Model ASR & OCR được chốt theo Phương án A
ASR_MODEL_ID = "vinai/PhoWhisper-small"
OCR_MODEL_FAMILY = "easyocr"  # Hoặc 'easy_vietocr', 'paddleocr'

# Ngưỡng lọc nhiễu & ROI
MIN_OCR_CONFIDENCE = 0.35  # Ngưỡng tin cậy OCR để cắt bỏ noise
USE_ROI_FILTERING = True   # Bật lọc ROI vùng tiêu đề/chữ động (Top/Bottom)

# Cấu hình ROI (tỷ lệ khung hình [y_min, y_max])
# Thường chữ tiêu đề/ticker nằm ở 25% đầu và 30% cuối màn hình
ROI_PROFILES = {
    "DEFAULT": {
        "top_banner": [0.0, 0.25],
        "bottom_ticker": [0.70, 1.0],
        "full_frame": [0.0, 1.0]
    }
}

# Tên file Parquet kết quả
OCR_PARQUET_PATH = OUTPUT_DIR / "ocr.parquet"
ASR_PARQUET_PATH = OUTPUT_DIR / "asr.parquet"
OCR_ASR_PARQUET_PATH = OUTPUT_DIR / "ocr_asr.parquet"
