"""
OCR Service - Hệ thống nhận diện chữ trên Keyframe (Optical Character Recognition):
  - 100% Cache-First: Tra cứu siêu tốc (< 0.0001s) từ file ocr_transcripts.json đã quét trước.
  - Tuyệt đối không làm đơ hay nghẽn Server khi người dùng click vào ảnh trên Web.
"""
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Danh sách nơi tìm file OCR JSON đã quét sẵn
OCR_JSON_CANDIDATES = [
    Path("D:/uploads/ocr_transcripts.json"),
    PROJECT_ROOT / "uploads" / "ocr_transcripts.json",
    PROJECT_ROOT / "ocr_transcripts.json",
]

_cached_ocr_db: Optional[Dict[str, Any]] = None


def _load_ocr_json() -> Dict[str, Any]:
    """Tải và lưu vào bộ nhớ RAM toàn bộ dữ liệu OCR từ JSON đã tiền xử lý"""
    global _cached_ocr_db
    if _cached_ocr_db is not None:
        return _cached_ocr_db

    for p in OCR_JSON_CANDIDATES:
        if p.is_file() and p.stat().st_size > 5:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _cached_ocr_db = json.load(f)
                print(f"✅ [OCR] Đã nạp thành công cache chữ OCR từ {p.name} ({len(_cached_ocr_db)} video).", flush=True)
                return _cached_ocr_db
            except Exception as e:
                print(f"⚠️ [OCR] Lỗi đọc file {p}: {e}", flush=True)

    _cached_ocr_db = {}
    return _cached_ocr_db


def get_ocr_text_for_frame(video_id: str, frame_str: str) -> str:
    """
    Tra cứu chữ OCR cho frame cụ thể trong < 0.0001 giây:
    Ví dụ: video_id="L21_V001", frame_str="001" hoặc "1" hoặc "079"
    """
    ocr_db = _load_ocr_json()
    v_stem = Path(video_id.strip()).stem
    f_clean = str(int(Path(frame_str).stem)) if Path(frame_str).stem.isdigit() else Path(frame_str).stem
    f_padded = f"{int(f_clean):03d}" if f_clean.isdigit() else f_clean

    # Kiểm tra trong OCR DB
    if v_stem in ocr_db:
        video_entry = ocr_db[v_stem]
        if isinstance(video_entry, dict):
            # Thử các định dạng key: "001", "1", "001.jpg", "1.jpg"
            for k in [f_padded, f_clean, f"{f_padded}.jpg", f"{f_clean}.jpg"]:
                if k in video_entry:
                    val = video_entry[k]
                    return val if isinstance(val, str) else val.get("text", "")

    return "ℹ️ Không có văn bản (hoặc chưa chạy batch OCR cho frame này)."


def process_image(image_path: str) -> dict:
    """
    Hàm giao tiếp với server API /api/details:
    - Trích xuất video_id và frame từ image_path
    - Tra cứu tức thì từ cache JSON
    """
    clean_p = image_path.replace("\\", "/").strip("/")
    parts = clean_p.split("/")
    
    if len(parts) >= 2:
        video_id = parts[-2]
        frame_name = parts[-1]
    else:
        video_id = Path(clean_p).stem.split("_")[0] if "_" in Path(clean_p).stem else Path(clean_p).stem
        frame_name = Path(clean_p).name

    text = get_ocr_text_for_frame(video_id, frame_name)
    return {
        "extracted_text": text,
        "description": f"Video {video_id} Frame {frame_name}"
    }