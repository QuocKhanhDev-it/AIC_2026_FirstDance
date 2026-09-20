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
import os
from pathlib import Path

# Đã xóa load DLL động ở đầu file

from typing import Any, List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image

try:
    import easyocr
except (ImportError, OSError):
    easyocr = None

try:
    from rapidocr_onnxruntime import RapidOCR
except (ImportError, OSError):
    RapidOCR = None

from pipeline_OCR_ASR.config import MIN_OCR_CONFIDENCE, OCR_PARQUET_PATH, USE_ROI_FILTERING, ROI_PROFILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


_KEYFRAME_DISK_CACHE: Dict[Tuple[str, str], Path] = {}


def build_disk_keyframe_cache(base_dir: Path = Path("D:/Study/AICChallenge")) -> Dict[Tuple[str, str], Path]:
    """Tạo chỉ mục bộ nhớ đệm O(1) cho toàn bộ ảnh keyframe trên đĩa, hỗ trợ cả 001.jpg, 0001.jpg, 1.jpg."""
    global _KEYFRAME_DISK_CACHE
    if _KEYFRAME_DISK_CACHE:
        return _KEYFRAME_DISK_CACHE

    logging.info("Đang quét chỉ mục bộ nhớ đệm cho 100% ảnh keyframe trên đĩa...")
    cache = {}
    for root, dirs, files in os.walk(base_dir):
        if ".zip" in root:
            continue
        vid = os.path.basename(root)
        if vid.startswith("L") and "_V" in vid:
            r_path = Path(root)
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    fname = f.lower()
                    stem = fname.split(".")[0]
                    cache[(vid, fname)] = r_path / f
                    if stem.isdigit():
                        cache[(vid, str(int(stem)))] = r_path / f
    _KEYFRAME_DISK_CACHE = cache
    logging.info("Đã lập chỉ mục xong %d vị trí ảnh keyframe thực tế.", len(_KEYFRAME_DISK_CACHE))
    return _KEYFRAME_DISK_CACHE


def find_keyframe_image(row: pd.Series) -> Optional[Path]:
    """Tìm file ảnh keyframe thực tế trên đĩa kể cả khi đường dẫn tuyệt đối trong master.parquet bị lệch."""
    kf_path_str = row.get("kf_path", "")
    if kf_path_str and pd.notna(kf_path_str):
        p = Path(kf_path_str)
        if p.exists():
            return p

    vid = str(row.get("video_id", ""))
    if not vid:
        return None

    kf_n = str(row.get("kf_n", ""))
    kf_name = str(row.get("kf_name", "")).lower() if pd.notna(row.get("kf_name")) else f"{kf_n}.jpg"

    cache = build_disk_keyframe_cache()
    return cache.get((vid, kf_name)) or cache.get((vid, kf_n)) or cache.get((vid, f"{kf_n}.jpg"))


import threading

_THREAD_LOCAL_READER = threading.local()


def get_thread_local_reader(languages: List[str] = ["vi", "en"], gpu: bool = True) -> Tuple[str, Any]:
    """Khởi tạo và lưu cache reader riêng cho từng worker thread trong ThreadPoolExecutor."""
    if not hasattr(_THREAD_LOCAL_READER, "reader") or _THREAD_LOCAL_READER.reader is None:
        if easyocr is not None:
            try:
                r = easyocr.Reader(languages, gpu=gpu)
                _THREAD_LOCAL_READER.reader = ("easyocr", r)
                return _THREAD_LOCAL_READER.reader
            except Exception:
                pass

        if RapidOCR is not None:
            # Khởi tạo với DirectML để ép chạy GPU qua DirectX 12
            r = RapidOCR(det_use_dml=gpu, cls_use_dml=gpu, rec_use_dml=gpu)
            
            # CẬP NHẬT RÀNG BUỘC THEO YÊU CẦU:
            if gpu:
                active_providers = r.text_det.infer.session.get_providers()
                if 'DmlExecutionProvider' not in active_providers:
                    raise RuntimeError(f"GPU KHÔNG CHẠY! ONNXRuntime đã ngắt GPU và nhảy về {active_providers}. Tiến trình dừng theo yêu cầu của bạn để tránh treo máy CPU.")

            _THREAD_LOCAL_READER.reader = ("rapidocr", r)
            return _THREAD_LOCAL_READER.reader

        raise RuntimeError("Không thể khởi tạo mô hình OCR (EasyOCR hoặc RapidOCR).")
    return _THREAD_LOCAL_READER.reader


# Biến toàn cục cho từng worker process
_WORKER_ENGINE = None
_WORKER_ENGINE_TYPE = None
_MIN_CONFIDENCE = 0.5

def init_worker(gpu: bool, languages: List[str], min_conf: float):
    global _WORKER_ENGINE, _WORKER_ENGINE_TYPE, _MIN_CONFIDENCE
    _MIN_CONFIDENCE = min_conf
    # Khởi tạo model 1 lần duy nhất cho process này
    _WORKER_ENGINE_TYPE, _WORKER_ENGINE = get_thread_local_reader(languages, gpu)

def process_single_record(record: dict) -> dict:
    global _WORKER_ENGINE, _WORKER_ENGINE_TYPE, _MIN_CONFIDENCE
    p = record.get("resolved_kf_path")
    if not p or not isinstance(p, Path) or not p.exists():
        return None

    try:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        top_crop = np.array(img.crop((0, 0, w, max(1, int(h * 0.35)))))
        bot_crop = np.array(img.crop((0, min(h - 1, int(h * 0.65)), w, h)))
    except Exception:
        return None

    valid_texts = []
    confidences = []
    raw_results = []
    
    try:
        if _WORKER_ENGINE_TYPE == "easyocr":
            res1 = _WORKER_ENGINE.readtext(top_crop, detail=1)
            res2 = _WORKER_ENGINE.readtext(bot_crop, detail=1)
            raw_results = (res1 or []) + (res2 or [])
        elif _WORKER_ENGINE_TYPE == "rapidocr":
            res1, _ = _WORKER_ENGINE(top_crop)
            res2, _ = _WORKER_ENGINE(bot_crop)
            full = (res1 or []) + (res2 or [])
            raw_results = [(it[0], it[1], float(it[2])) for it in full] if full else []
    except Exception as e:
        logging.warning("Lỗi OCR row %s: %s", record.get("row_id"), e)

    for res in raw_results:
        if len(res) >= 3:
            text, conf = str(res[1]).strip(), float(res[2])
            if conf >= _MIN_CONFIDENCE and len(text) >= 2:
                valid_texts.append(text)
                confidences.append(conf)

    ocr_text = " ".join(valid_texts)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "row_id": int(record["row_id"]),
        "video_id": str(record["video_id"]),
        "kf_n": int(record["kf_n"]),
        "ocr_text": ocr_text,
        "confidence": round(mean_conf, 4),
        "boxes_count": len(valid_texts)
    }

class KeyframeOCRProcessor:
    def __init__(self, languages: List[str] = ["vi", "en"], gpu: bool = True, min_confidence: float = MIN_OCR_CONFIDENCE):
        self.min_confidence = min_confidence
        self.gpu = gpu
        self.languages = languages

    def extract_text_from_image(self, image_path: Path, use_roi: bool = USE_ROI_FILTERING) -> Dict[str, Any]:
        """Trích xuất chữ từ 1 ảnh keyframe."""
        if not image_path.exists():
            return {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
        except Exception as e:
            logging.warning("Không thể đọc ảnh %s: %s", image_path, e)
            return {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

        engine_type, engine = get_thread_local_reader(self.languages, self.gpu)
        raw_results = []

        def run_ocr_on_pil(pil_img):
            arr = np.array(pil_img)
            if engine_type == "easyocr":
                return engine.readtext(arr, detail=1)
            elif engine_type == "rapidocr":
                res, _ = engine(arr)
                return [(item[0], item[1], float(item[2])) for item in res] if res else []
            return []

        try:
            if use_roi:
                top_img = img.crop((0, 0, w, max(1, int(h * 0.35))))
                bot_img = img.crop((0, min(h - 1, int(h * 0.65)), w, h))
                raw_results = run_ocr_on_pil(top_img) + run_ocr_on_pil(bot_img)
            else:
                raw_results = run_ocr_on_pil(img)
        except Exception as e:
            logging.warning("Lỗi OCR ảnh %s: %s", image_path, e)
            return {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

        valid_texts = []
        confidences = []

        for item in raw_results:
            if len(item) < 3:
                continue
            bbox, text, conf = item[0], item[1], float(item[2])
            if conf < self.min_confidence:
                continue

            text_clean = str(text).strip()
            if not text_clean or len(text_clean) < 2:
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

    def process_dataset(self, df_master: pd.DataFrame, limit: Optional[int] = None, output_path: Path = OCR_PARQUET_PATH, batch_size: int = 500, max_workers: int = 4) -> pd.DataFrame:
        """Chạy OCR sản xuất với Queue Pipeline TĂNG TỐC TỐI ƯU GPU và LƯU CHECKPOINT TỰ ĐỘNG."""
        import queue
        import threading
        import time

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Nạp Checkpoint đã có để tiếp tục (Resume) nếu bị ngắt giữa chừng
        existing_rows: list[dict[str, Any]] = []
        processed_rids = set()

        if output_path.exists() and output_path.stat().st_size > 0:
            try:
                df_existing = pd.read_parquet(output_path)
                if not df_existing.empty and "row_id" in df_existing.columns:
                    existing_rows = df_existing.to_dict("records")
                    processed_rids = set(df_existing["row_id"].unique())
                    logging.info("📌 TỰ ĐỘNG NẠP CHECKPOINT: Đã có %d / %d keyframe trong file %s.", len(processed_rids), len(df_master), output_path.name)
            except Exception as e:
                logging.warning("Không thể nạp checkpoint từ %s (%s). Sẽ tạo mới...", output_path, e)

        # 2. Định vị file ảnh thực tế trên đĩa (Fast disk index)
        logging.info("Đang kiểm tra vị trí ảnh keyframe thực tế trên đĩa...")
        cache = build_disk_keyframe_cache()
        resolved_paths = []

        for r in df_master.itertuples():
            vid = str(r.video_id)
            kf_n = str(r.kf_n)
            kf_name = str(r.kf_name).lower() if pd.notna(r.kf_name) and r.kf_name else f"{kf_n}.jpg"
            p = cache.get((vid, kf_name)) or cache.get((vid, kf_n)) or cache.get((vid, f"{kf_n}.jpg"))
            resolved_paths.append(p)

        df_master = df_master.copy()
        df_master["resolved_kf_path"] = resolved_paths

        df_valid = df_master[df_master["resolved_kf_path"].notna()]
        logging.info("Tìm thấy %d / %d keyframe tồn tại trên đĩa.", len(df_valid), len(df_master))

        # Áp dụng limit nếu có
        df_target = df_valid.head(limit) if limit else df_valid

        # Lọc bỏ các row_id đã được xử lý trong checkpoint
        df_todo = df_target[~df_target["row_id"].isin(processed_rids)]
        total_todo = len(df_todo)

        if total_todo == 0:
            logging.info("✅ HOÀN TẤT! Tất cả %d keyframe mục tiêu đã được xử lý trong checkpoint.", len(df_target))
            return pd.DataFrame(existing_rows)

        num_consumers = max(2, min(6, max_workers))
        logging.info("⚡ BẮT ĐẦU CHẠY PIPELINE OCR (GPU ONNX + %d Consumer Processes) cho %d keyframe...", num_consumers, total_todo)

        all_rows = list(existing_rows)
        processed_count = 0
        t_start = time.time()
        
        def format_time(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hrs > 0:
                return f"{hrs:02d}h {mins:02d}m {secs:02d}s"
            return f"{mins:02d}m {secs:02d}s"

        import multiprocessing as mp
        # Dùng spawn context để tương thích tốt nhất với Windows
        ctx = mp.get_context("spawn")
        
        todo_records = df_todo.to_dict("records")
        
        with ctx.Pool(processes=num_consumers, initializer=init_worker, initargs=(self.gpu, self.languages, self.min_confidence)) as pool:
            for result in pool.imap_unordered(process_single_record, todo_records):
                if result is not None:
                    all_rows.append(result)
                
                processed_count += 1
                
                if processed_count % 50 == 0 or processed_count == total_todo:
                    elapsed_total = time.time() - t_start
                    curr_speed = processed_count / elapsed_total if elapsed_total > 0 else 0.0
                    rem_items = total_todo - processed_count
                    eta_sec = rem_items / curr_speed if curr_speed > 0 else 0.0
                    pct = (len(all_rows) / len(df_master)) * 100

                    print(f"\r[OCR Dashboard] 🚀 Progress: {len(all_rows):,}/{len(df_master):,} ({pct:.2f}%) | "
                          f"Speed: {curr_speed:.1f} kf/s | Elapsed: {format_time(elapsed_total)} | "
                          f"ETA Remaining: {format_time(eta_sec)}", end="", flush=True)

                if processed_count % batch_size == 0 or processed_count == total_todo:
                    print()
                    df_current = pd.DataFrame(all_rows)
                    df_current.to_parquet(output_path, index=False)
                    logging.info("💾 Checkpoint saved! [%d / %d keyframes]", len(all_rows), len(df_master))

        total_time = time.time() - t_start
        logging.info("🎉 HOÀN TẤT TOÀN BỘ OCR! Tổng số dòng: %d | Thời gian: %s", len(all_rows), format_time(total_time))
        return pd.DataFrame(all_rows)
