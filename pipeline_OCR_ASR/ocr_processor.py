"""
ocr_processor.py — Module xử lý trích xuất văn bản từ Keyframe (OCR Production)

Hỗ trợ:
1. Đọc keyframe ảnh từ đường dẫn hoặc theo row_id trong master.parquet.
2. Quét OCR toàn khung hình (Full-frame) để bắt trọn mọi chữ (biển hiệu, biển số, slide, logo).
3. Hỗ trợ tùy chọn cắt dải ROI khi cần tập trung vào banner/ticker.
4. Lọc theo ngưỡng tin cậy (confidence gate >= 0.5) và lọc nhiễu ký tự đơn lẻ.
5. Xuất dữ liệu ra file ocr.parquet chuẩn hóa với cơ chế lưu Checkpoint nguyên tử (Atomic Write).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
except (ImportError, OSError):
    RapidOCR = None

try:
    import easyocr
except (ImportError, OSError):
    easyocr = None

from pipeline_OCR_ASR.config import (
    DATA_DIR,
    MIN_OCR_CONFIDENCE,
    OCR_PARQUET_PATH,
    USE_ROI_FILTERING,
    OCR_ENGINE_PREFERENCE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


_KEYFRAME_DISK_CACHE: Dict[Tuple[str, str], Path] = {}


def build_disk_keyframe_cache(base_dir: Optional[Path] = None) -> Dict[Tuple[str, str], Path]:
    """Tạo chỉ mục bộ nhớ đệm O(1) cho toàn bộ ảnh keyframe trên đĩa."""
    global _KEYFRAME_DISK_CACHE
    if _KEYFRAME_DISK_CACHE:
        return _KEYFRAME_DISK_CACHE

    target_dir = base_dir if base_dir and base_dir.exists() else DATA_DIR
    logging.info("Đang quét chỉ mục bộ nhớ đệm cho ảnh keyframe tại %s...", target_dir)
    cache = {}
    for root, dirs, files in os.walk(target_dir):
        if ".zip" in root or "__MACOSX" in root:
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
                        cache[(vid, f"{int(stem)}.jpg")] = r_path / f
                        cache[(vid, f"{int(stem):03d}.jpg")] = r_path / f
                        cache[(vid, f"{int(stem):04d}.jpg")] = r_path / f
    _KEYFRAME_DISK_CACHE = cache
    logging.info("Đã lập chỉ mục xong %d vị trí ảnh keyframe thực tế.", len(_KEYFRAME_DISK_CACHE))
    return _KEYFRAME_DISK_CACHE


def find_keyframe_image(row: pd.Series, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Tìm file ảnh keyframe thực tế trên đĩa."""
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

    cache = build_disk_keyframe_cache(base_dir)
    return cache.get((vid, kf_name)) or cache.get((vid, kf_n)) or cache.get((vid, f"{kf_n}.jpg"))


def create_ocr_engine(languages: List[str] = ["vi", "en"], gpu: bool = True, preference: str = OCR_ENGINE_PREFERENCE) -> Tuple[str, Any]:
    """Khởi tạo engine OCR theo thứ tự ưu tiên."""
    if preference == "rapidocr" and RapidOCR is not None:
        try:
            # Khởi tạo RapidOCR với DirectML nếu bật gpu
            engine = RapidOCR(det_use_dml=gpu, cls_use_dml=gpu, rec_use_dml=gpu)
            return "rapidocr", engine
        except Exception as e:
            logging.warning("Không thể khởi tạo RapidOCR GPU (%s). Thử khởi tạo CPU...", e)
            try:
                engine = RapidOCR(det_use_dml=False, cls_use_dml=False, rec_use_dml=False)
                return "rapidocr", engine
            except Exception as e_cpu:
                logging.warning("RapidOCR CPU thất bại: %s", e_cpu)

    if easyocr is not None:
        try:
            engine = easyocr.Reader(languages, gpu=gpu)
            return "easyocr", engine
        except Exception as e:
            logging.warning("EasyOCR thất bại: %s", e)

    if RapidOCR is not None:
        engine = RapidOCR(det_use_dml=False, cls_use_dml=False, rec_use_dml=False)
        return "rapidocr", engine

    raise RuntimeError("Không thể khởi tạo bất kỳ mô hình OCR nào (cần RapidOCR hoặc EasyOCR).")


# Biến toàn cục cho từng worker process
_WORKER_ENGINE = None
_WORKER_ENGINE_TYPE = None
_MIN_CONFIDENCE = 0.50
_USE_ROI = False


def init_worker(gpu: bool, languages: List[str], min_conf: float, use_roi: bool, preference: str):
    global _WORKER_ENGINE, _WORKER_ENGINE_TYPE, _MIN_CONFIDENCE, _USE_ROI
    _MIN_CONFIDENCE = min_conf
    _USE_ROI = use_roi
    _WORKER_ENGINE_TYPE, _WORKER_ENGINE = create_ocr_engine(languages=languages, gpu=gpu, preference=preference)


def process_single_record(record: dict) -> Optional[dict]:
    global _WORKER_ENGINE, _WORKER_ENGINE_TYPE, _MIN_CONFIDENCE, _USE_ROI
    p = record.get("resolved_kf_path")
    if not p or not isinstance(p, Path) or not p.exists():
        return None

    try:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        crops = []
        if _USE_ROI:
            top_crop = np.array(img.crop((0, 0, w, max(1, int(h * 0.35)))))
            bot_crop = np.array(img.crop((0, min(h - 1, int(h * 0.65)), w, h)))
            crops = [top_crop, bot_crop]
        else:
            # Mặc định quét full frame để bắt trọn mọi chữ
            crops = [np.array(img)]
    except Exception:
        return None

    raw_results = []
    try:
        if _WORKER_ENGINE_TYPE == "rapidocr":
            for crop_arr in crops:
                res, _ = _WORKER_ENGINE(crop_arr)
                if res:
                    raw_results.extend([(it[0], it[1], float(it[2])) for it in res])
        elif _WORKER_ENGINE_TYPE == "easyocr":
            for crop_arr in crops:
                res = _WORKER_ENGINE.readtext(crop_arr, detail=1)
                if res:
                    raw_results.extend(res)
    except Exception as e:
        logging.warning("Lỗi OCR row %s: %s", record.get("row_id"), e)

    valid_texts = []
    confidences = []

    for res in raw_results:
        if len(res) >= 3:
            text = str(res[1]).strip()
            conf = float(res[2])
            # Lọc bỏ rác ký tự đơn lẻ hoặc độ tin cậy thấp
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
    def __init__(self, languages: List[str] = ["vi", "en"], gpu: bool = True,
                 min_confidence: float = MIN_OCR_CONFIDENCE,
                 use_roi: bool = USE_ROI_FILTERING,
                 engine_preference: str = OCR_ENGINE_PREFERENCE):
        self.min_confidence = min_confidence
        self.gpu = gpu
        self.languages = languages
        self.use_roi = use_roi
        self.engine_preference = engine_preference
        self._local_engine = None
        self._local_engine_type = None

    def _get_engine(self):
        if self._local_engine is None:
            self._local_engine_type, self._local_engine = create_ocr_engine(
                languages=self.languages, gpu=self.gpu, preference=self.engine_preference
            )
        return self._local_engine_type, self._local_engine

    def extract_text_from_image(self, image_path: Path, use_roi: Optional[bool] = None) -> Dict[str, Any]:
        """Trích xuất chữ từ 1 ảnh keyframe."""
        if not image_path.exists():
            return {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

        roi_flag = self.use_roi if use_roi is None else use_roi
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            if roi_flag:
                crops = [
                    np.array(img.crop((0, 0, w, max(1, int(h * 0.35))))),
                    np.array(img.crop((0, min(h - 1, int(h * 0.65)), w, h)))
                ]
            else:
                crops = [np.array(img)]
        except Exception as e:
            logging.warning("Không thể đọc ảnh %s: %s", image_path, e)
            return {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

        engine_type, engine = self._get_engine()
        raw_results = []

        try:
            if engine_type == "rapidocr":
                for crop in crops:
                    res, _ = engine(crop)
                    if res:
                        raw_results.extend([(it[0], it[1], float(it[2])) for it in res])
            elif engine_type == "easyocr":
                for crop in crops:
                    res = engine.readtext(crop, detail=1)
                    if res:
                        raw_results.extend(res)
        except Exception as e:
            logging.warning("Lỗi OCR ảnh %s: %s", image_path, e)
            return {"ocr_text": "", "confidence": 0.0, "boxes_count": 0, "texts": []}

        valid_texts = []
        confidences = []

        for item in raw_results:
            if len(item) < 3:
                continue
            text, conf = str(item[1]).strip(), float(item[2])
            if conf >= self.min_confidence and len(text) >= 2:
                valid_texts.append(text)
                confidences.append(conf)

        combined_text = " ".join(valid_texts)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "ocr_text": combined_text,
            "confidence": round(mean_conf, 4),
            "boxes_count": len(valid_texts),
            "texts": valid_texts
        }

    def process_dataset(self, df_master: pd.DataFrame, limit: Optional[int] = None,
                        output_path: Path = OCR_PARQUET_PATH,
                        base_dir: Optional[Path] = None,
                        batch_size: int = 500, max_workers: int = 6,
                        overwrite: bool = False,
                        reprocess_up_to: Optional[int] = None) -> pd.DataFrame:
        """Chạy OCR sản xuất với Multiprocessing, Atomic Checkpoint và hỗ trợ Reprocess mốc chỉ định."""
        import multiprocessing as mp
        import time

        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_rows: list[dict[str, Any]] = []
        processed_rids = set()

        if overwrite:
            if output_path.exists() and output_path.stat().st_size > 0:
                bak_path = output_path.with_suffix(".bak.parquet")
                try:
                    import shutil
                    shutil.copy2(output_path, bak_path)
                    logging.info("🔄 ĐÃ BẬT OVERWRITE: Đã sao lưu file cũ sang %s và sẽ chạy mới 100%% với logic Full-frame.", bak_path.name)
                except Exception as e:
                    logging.warning("Không thể sao lưu file cũ: %s", e)
        elif reprocess_up_to is not None:
            if output_path.exists() and output_path.stat().st_size > 0:
                try:
                    df_existing = pd.read_parquet(output_path)
                    if not df_existing.empty and "row_id" in df_existing.columns:
                        # Giữ lại các keyframe vượt mốc reprocess_up_to (đã chuẩn)
                        df_keep = df_existing[df_existing["row_id"] > reprocess_up_to]
                        existing_rows = df_keep.to_dict("records")
                        # Đánh dấu các row_id cần giữ để không chạy lại
                        processed_rids = set(df_keep["row_id"].unique())
                        logging.info("🎯 CHẾ ĐỘ REPROCESS MỐC: Giữ nguyên %d keyframe (row_id > %d) và sẽ overwrite các keyframe <= %d.",
                                     len(df_keep), reprocess_up_to, reprocess_up_to)
                except Exception as e:
                    logging.warning("Không thể nạp file để reprocess: %s", e)
        else:
            if output_path.exists() and output_path.stat().st_size > 0:
                try:
                    df_existing = pd.read_parquet(output_path)
                    if not df_existing.empty and "row_id" in df_existing.columns:
                        existing_rows = df_existing.to_dict("records")
                        processed_rids = set(df_existing["row_id"].unique())
                        logging.info("📌 TỰ ĐỘNG NẠP CHECKPOINT: Đã có %d / %d keyframe trong file %s.",
                                     len(processed_rids), len(df_master), output_path.name)
                except Exception as e:
                    logging.warning("Không thể nạp checkpoint từ %s (%s). Sẽ tạo mới...", output_path, e)

        # 2. Định vị file ảnh thực tế trên đĩa
        logging.info("Đang kiểm tra vị trí ảnh keyframe thực tế trên đĩa...")
        cache = build_disk_keyframe_cache(base_dir)
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

        # Áp dụng limit hoặc phạm vi reprocess_up_to
        if reprocess_up_to is not None:
            df_target = df_valid[df_valid["row_id"] <= reprocess_up_to]
            if limit:
                df_target = df_target.head(limit)
        else:
            df_target = df_valid.head(limit) if limit else df_valid

        # Lọc bỏ các row_id không cần xử lý
        df_todo = df_target[~df_target["row_id"].isin(processed_rids)]
        total_todo = len(df_todo)

        if total_todo == 0:
            logging.info("✅ HOÀN TẤT! Tất cả %d keyframe mục tiêu đã được xử lý trong checkpoint.", len(df_target))
            return pd.DataFrame(existing_rows)

        num_workers = max(1, min(12, max_workers))
        logging.info("⚡ BẮT ĐẦU CHẠY PIPELINE OCR (Engine: %s, Workers: %d, Full-frame: %s) cho %d keyframe...",
                     self.engine_preference, num_workers, not self.use_roi, total_todo)

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

        ctx = mp.get_context("spawn")
        todo_records = df_todo.to_dict("records")

        with ctx.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(self.gpu, self.languages, self.min_confidence, self.use_roi, self.engine_preference)
        ) as pool:
            for result in pool.imap_unordered(process_single_record, todo_records, chunksize=16):
                if result is not None:
                    all_rows.append(result)

                processed_count += 1
                if processed_count % 50 == 0 or processed_count == total_todo:
                    elapsed_total = time.time() - t_start
                    curr_speed = processed_count / elapsed_total if elapsed_total > 0 else 0.0
                    rem_items = total_todo - processed_count
                    eta_sec = rem_items / curr_speed if curr_speed > 0 else 0.0
                    pct = (len(all_rows) / len(df_master)) * 100

                    print(f"\r[OCR Progress] 🚀 {len(all_rows):,}/{len(df_master):,} ({pct:.2f}%) | "
                          f"Speed: {curr_speed:.1f} kf/s | Elapsed: {format_time(elapsed_total)} | "
                          f"ETA: {format_time(eta_sec)}", end="", flush=True)

                if processed_count % batch_size == 0 or processed_count == total_todo:
                    print()
                    df_current = pd.DataFrame(all_rows).drop_duplicates(subset=["row_id"], keep="last").sort_values("row_id")
                    tmp_file = output_path.with_suffix(".tmp.parquet")
                    df_current.to_parquet(tmp_file, index=False)
                    tmp_file.replace(output_path)
                    logging.info("💾 Checkpoint đã lưu an toàn! [%d / %d keyframes]", len(df_current), len(df_master))

        total_time = time.time() - t_start
        df_final = pd.DataFrame(all_rows).drop_duplicates(subset=["row_id"], keep="last").sort_values("row_id")
        logging.info("🎉 HOÀN TẤT TOÀN BỘ OCR! Tổng số dòng: %d | Thời gian: %s", len(df_final), format_time(total_time))
        return df_final
