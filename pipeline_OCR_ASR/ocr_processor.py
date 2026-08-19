"""
ocr_processor.py — Module xử lý trích xuất văn bản từ Keyframe (OCR Production)

Hỗ trợ:
1. Đọc keyframe ảnh từ đường dẫn hoặc theo row_id trong master.parquet.
2. Áp dụng ROI (Region of Interest) để loại bỏ 80-90% false-positive nền.
3. Lọc theo ngưỡng tin cậy (confidence gate).
4. Xuất dữ liệu ra file ocr.parquet chuẩn hóa.
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Dict, Optional
import pandas as pd
import numpy as np
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None

from pipeline_OCR_ASR.config import MIN_OCR_CONFIDENCE, OCR_PARQUET_PATH, USE_ROI_FILTERING, ROI_PROFILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class KeyframeOCRProcessor:
    def __init__(self, languages: List[str] = ["vi", "en"], gpu: bool = True, min_confidence: float = MIN_OCR_CONFIDENCE):
        self.min_confidence = min_confidence
        self.gpu = gpu
        self.languages = languages
        self.reader = None

    def _init_reader(self):
        if self.reader is None:
            if easyocr is None:
                raise RuntimeError("Thư viện easyocr chưa được cài đặt.")
            logging.info("Đang khởi tạo EasyOCR reader (languages=%s, gpu=%s)...", self.languages, self.gpu)
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)

    def extract_text_from_image(self, image_path: Path, use_roi: bool = USE_ROI_FILTERING) -> Dict[str, Any]:
        """Trích xuất chữ từ 1 ảnh keyframe.
        
        Nếu use_roi=True, sẽ tập trung trích chữ từ vùng Banner trên (Header)
        và Ticker dưới (Footer) để giảm false positive từ cảnh vật trung tâm.
        """
        self._init_reader()
        if not image_path.exists():
            return {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
        except Exception as e:
            logging.warning("Không thể đọc ảnh %s: %s", image_path, e)
            return {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

        raw_results = self.reader.readtext(str(image_path), detail=1)
        valid_texts = []
        confidences = []

        for bbox, text, conf in raw_results:
            if conf < self.min_confidence:
                continue

            text_clean = str(text).strip()
            if not text_clean or len(text_clean) < 2:
                continue

            if use_roi:
                # bbox dạng [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                ys = [pt[1] for pt in bbox]
                avg_y = sum(ys) / (len(ys) * h) if h > 0 else 0.5
                # Kiểm tra nếu nằm trong ROI top (<= 0.35) hoặc bottom (>= 0.65)
                is_in_roi = (avg_y <= 0.35) or (avg_y >= 0.65)
                if not is_in_roi:
                    continue

            valid_texts.append(text_clean)
            confidences.append(float(conf))

        combined_text = " ".join(valid_texts)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "ocr_text": combined_text,
            "confidence": round(mean_conf, 4),
            "boxes_count": len(valid_texts),
            "texts": valid_texts
        }

    def process_dataset(self, df_master: pd.DataFrame, limit: Optional[int] = None, output_path: Path = OCR_PARQUET_PATH, only_existing: bool = True) -> pd.DataFrame:
        """Chạy OCR sản xuất trên tập hợp keyframe trong df_master."""
        self._init_reader()
        
        # Nếu filter chỉ các keyframe tồn tại thực tế trên đĩa
        if only_existing and "kf_path" in df_master.columns:
            import os
            df_valid = df_master[df_master["kf_path"].apply(lambda p: os.path.exists(p) if pd.notna(p) and p else False)]
            if not df_valid.empty:
                logging.info("Tìm thấy %d / %d keyframe tồn tại trên đĩa.", len(df_valid), len(df_master))
                df_subset = df_valid.head(limit) if limit else df_valid
            else:
                df_subset = df_master.head(limit) if limit else df_master
        else:
            df_subset = df_master.head(limit) if limit else df_master

        rows = []
        total = len(df_subset)
        logging.info("Bắt đầu chạy OCR production cho %d keyframe...", total)

        for i, (_, row) in enumerate(df_subset.iterrows()):
            kf_path_str = row.get("kf_path", "")
            kf_path = Path(kf_path_str) if kf_path_str and pd.notna(kf_path_str) else None

            if kf_path and kf_path.exists():
                res = self.extract_text_from_image(kf_path)
            else:
                res = {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

            rows.append({
                "row_id": int(row["row_id"]),
                "video_id": str(row["video_id"]),
                "kf_n": int(row["kf_n"]),
                "ocr_text": res["ocr_text"],
                "confidence": res["confidence"],
                "boxes_count": res["boxes_count"]
            })

            if (i + 1) % 50 == 0 or (i + 1) == total:
                logging.info("Đã xử lý %d / %d keyframe...", i + 1, total)

        df_ocr = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_ocr.to_parquet(output_path, index=False)
        logging.info("Đã xuất kết quả OCR ra file: %s (%d dòng)", output_path, len(df_ocr))
        return df_ocr
