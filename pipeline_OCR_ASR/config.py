"""
config.py — Cấu hình hệ thống OCR & ASR Production (Giai đoạn 1)
"""

import os
from pathlib import Path
import torch

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu chỉ mục & output
INDEX_DIR = BASE_DIR / "index"
OUTPUT_DIR = BASE_DIR / "pipeline_OCR_ASR" / "output"

# Tìm thư mục chứa dữ liệu video / keyframes
env_data_dir = os.environ.get("AIC_DATA_DIR", "").strip()
CANDIDATE_DATA_DIRS = [
    Path(env_data_dir) if env_data_dir else None,
    Path("D:/Study/AICChallenge"),
    Path("C:/Code/aic_data"),
    BASE_DIR / "data",
]
DATA_DIR = next((p for p in CANDIDATE_DATA_DIRS if p is not None and p.exists() and str(p) != "."), Path("D:/Study/AICChallenge"))

# Model ASR & OCR được chốt
ASR_MODEL_ID = "vinai/PhoWhisper-small"
ASR_LOCAL_CACHE = Path("D:/Library/ai_cache/PhoWhisper-small-ct2")
ASR_MODEL_PATH = str(ASR_LOCAL_CACHE if ASR_LOCAL_CACHE.exists() else ASR_MODEL_ID)

# Cấu hình thiết bị ASR (Ưu tiên CUDA nếu có GPU NVIDIA)
ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_COMPUTE_TYPE = "float16" if ASR_DEVICE == "cuda" else "int8"

# OCR Engine: Ưu tiên rapidocr (nhanh gấp 3-4x, ổn định), fallback sang easyocr
OCR_ENGINE_PREFERENCE = "rapidocr"  # 'rapidocr' | 'easyocr'

# Ngưỡng lọc nhiễu & ROI
MIN_OCR_CONFIDENCE = 0.50   # Ngưỡng tin cậy OCR để cắt bỏ noise
USE_ROI_FILTERING = False   # Mặc định False để quét TOÀN BỘ KHUNG HÌNH (tránh mất chữ ở giữa)

# Tên file Parquet kết quả
OCR_PARQUET_PATH = OUTPUT_DIR / "ocr.parquet"
ASR_PARQUET_PATH = OUTPUT_DIR / "asr.parquet"
OCR_ASR_PARQUET_PATH = OUTPUT_DIR / "ocr_asr.parquet"

