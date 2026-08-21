"""
run_production_pipeline.py — Script điều phối chạy OCR & ASR Production cho Giai đoạn 1

Sử dụng:
    # 1. Chạy OCR trên GPU / CPU
    $env:PYTHONIOENCODING="utf-8"
    .venv\Scripts\python.exe -m pipeline_OCR_ASR.run_production_pipeline --ocr --workers 6

    # 2. Chạy ASR trên GPU NVIDIA CUDA
    .venv\Scripts\python.exe -m pipeline_OCR_ASR.run_production_pipeline --asr --device cuda

    # 3. Gộp kết quả OCR và ASR thành ocr_asr.parquet
    .venv\Scripts\python.exe -m pipeline_OCR_ASR.run_production_pipeline --gop

    # 4. Thử nghiệm truy vấn kết hợp BM25 và Lọc cứng token hiếm
    .venv\Scripts\python.exe -m pipeline_OCR_ASR.run_production_pipeline --test-query "79A-12345"
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from pipeline_OCR_ASR.config import (
    BASE_DIR,
    INDEX_DIR,
    OUTPUT_DIR,
    OCR_PARQUET_PATH,
    ASR_PARQUET_PATH,
    OCR_ASR_PARQUET_PATH,
    DATA_DIR,
    ASR_DEVICE,
    OCR_ENGINE_PREFERENCE
)
from pipeline_OCR_ASR.ocr_processor import KeyframeOCRProcessor
from pipeline_OCR_ASR.gop_ocr_asr import gop_ocr_va_asr
from pipeline_OCR_ASR.loc_cung_token_hiem import LocCungTokenHiem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def dinh_dang_thoi_gian(giay: float) -> str:
    """Đổi số giây pts_time thành chuỗi mm:ss hoặc hh:mm:ss dễ đọc."""
    m = int(giay // 60)
    s = int(giay % 60)
    h = m // 60
    m = m % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Pipeline OCR & ASR Production - AIC 2026 Stage 1")
    parser.add_argument("--master", type=Path, default=INDEX_DIR / "master.parquet", help="Đường dẫn file master.parquet")
    parser.add_argument("--ocr", action="store_true", help="Chạy OCR production")
    parser.add_argument("--asr", action="store_true", help="Chạy ASR production")
    parser.add_argument("--overwrite", action="store_true", help="Chạy lại mới 100% từ đầu (ghi đè và sao lưu checkpoint cũ)")
    parser.add_argument("--reprocess-up-to", type=int, default=None, help="Chỉ overwrite/reprocess các keyframe có row_id <= mốc chỉ định (ví dụ: 63100)")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số keyframe xử lý (dành cho thử nghiệm)")
    parser.add_argument("--workers", type=int, default=4, help="Số lượng worker chạy song song cho OCR/ASR (khuyến nghị: 4)")
    parser.add_argument("--engine", type=str, default=OCR_ENGINE_PREFERENCE, choices=["rapidocr", "easyocr"], help="Engine OCR ưu tiên")
    parser.add_argument("--device", type=str, default=ASR_DEVICE, choices=["cuda", "cpu"], help="Thiết bị chạy ASR (cuda / cpu)")
    parser.add_argument("--beam-size", type=int, default=5, choices=[1, 2, 3, 4, 5], help="Beam size cho ASR (5: chuẩn xác nhất, 1: nhanh gấp 2.5 lần)")
    parser.add_argument("--roi", action="store_true", help="Bật cắt ROI Top/Bottom (mặc định tắt để quét toàn khung)")
    parser.add_argument("--gop", action="store_true", help="Gộp OCR và ASR thành ocr_asr.parquet")
    parser.add_argument("--test-query", type=str, default=None, help="Thử nghiệm câu truy vấn trên kênh OCR/ASR")
    parser.add_argument("--video-dir", type=Path, default=DATA_DIR, help="Thư mục chứa video/keyframes gốc")

    args = parser.parse_args()

    if not args.master.exists():
        logging.error("Không tìm thấy file master.parquet tại %s", args.master)
        return

    logging.info("Nạp file master.parquet từ %s...", args.master)
    df_master = pd.read_parquet(args.master)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chạy OCR
    if args.ocr:
        ocr_proc = KeyframeOCRProcessor(
            use_roi=args.roi,
            engine_preference=args.engine
        )
        ocr_proc.process_dataset(
            df_master,
            limit=args.limit,
            output_path=OCR_PARQUET_PATH,
            base_dir=args.video_dir,
            max_workers=args.workers,
            overwrite=args.overwrite,
            reprocess_up_to=args.reprocess_up_to
        )

    # 2. Chạy ASR
    if args.asr:
        from pipeline_OCR_ASR.asr_processor import VideoASRProcessor
        asr_proc = VideoASRProcessor(device=args.device, beam_size=args.beam_size)
        asr_proc.process_dataset(
            df_master,
            base_dir=args.video_dir,
            output_path=ASR_PARQUET_PATH,
            max_workers=args.workers
        )

    # 3. Gộp OCR + ASR
    if args.gop:
        df_ocr = pd.read_parquet(OCR_PARQUET_PATH) if OCR_PARQUET_PATH.exists() else None
        df_asr = pd.read_parquet(ASR_PARQUET_PATH) if ASR_PARQUET_PATH.exists() else None
        gop_ocr_va_asr(df_master, df_ocr, df_asr, output_path=OCR_ASR_PARQUET_PATH)

    # 4. Test query với BM25 & Lọc cứng token hiếm
    if args.test_query:
        if not OCR_ASR_PARQUET_PATH.exists():
            logging.warning("Chưa có file ocr_asr.parquet tại %s. Tiến hành gộp dữ liệu tạm...", OCR_ASR_PARQUET_PATH)
            df_ocr = pd.read_parquet(OCR_PARQUET_PATH) if OCR_PARQUET_PATH.exists() else None
            df_asr = pd.read_parquet(ASR_PARQUET_PATH) if ASR_PARQUET_PATH.exists() else None
            df_ocr_asr = gop_ocr_va_asr(df_master, df_ocr, df_asr, output_path=OCR_ASR_PARQUET_PATH)
        else:
            df_ocr_asr = pd.read_parquet(OCR_ASR_PARQUET_PATH)

        logging.info("=== THỬ NGHIỆM TRUY VẤN: '%s' ===", args.test_query)

        # Test BM25
        try:
            from src.bm25 import KenhVanBan
            kenh_ocr_asr = KenhVanBan.tu_bang_khung(df_master, df_ocr_asr, cot="text", ten="ocr_asr")
            res_bm25 = kenh_ocr_asr.tim(args.test_query, k=5)
            print("\n--- KẾT QUẢ BM25 (Kênh 3 OCR/ASR) ---")
            for c in res_bm25:
                pts = c.meta.get("pts_time", 0.0)
                time_str = dinh_dang_thoi_gian(pts)
                print(f"Score: {c.score:6.2f} | Video: {c.video_id:<10} | Time: {time_str:>7} ({pts:6.1f}s) | Frame: {c.frame_idx:>6} | Row: {c.row_id:>7} | Title: {str(c.meta['title'])[:35]}")
        except Exception as e:
            logging.error("Không thể chạy BM25 test: %s", e)

        # Test Lọc Cứng Token Hiếm
        loc_cung = LocCungTokenHiem(df_ocr_asr, df_master)
        res_loc_cung = loc_cung.tim_kiem_loc_cung(args.test_query, top_k=5)
        print("\n--- KẾT QUẢ LỌC CỨNG TOKEN HIẾM (WHERE text LIKE '%...%') ---")
        if res_loc_cung:
            for c in res_loc_cung:
                pts = c.meta.get("pts_time", 0.0)
                time_str = dinh_dang_thoi_gian(pts)
                print(f"Bonus: {c.score:6.2f} | Video: {c.video_id:<10} | Time: {time_str:>7} ({pts:6.1f}s) | Frame: {c.frame_idx:>6} | Row: {c.row_id:>7} | Matched: {c.meta['matched_text']}")
        else:
            print("Không có token hiếm nào khớp cứng.")


if __name__ == "__main__":
    main()

