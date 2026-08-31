"""
Script chạy Batch Offline OCR (PaddleOCR v4) độc lập trước ngày thi:
  - Quét toàn bộ ảnh keyframe trên D:\ và C:\
  - Bóc tách văn bản trong từng ảnh
  - Xuất ra file ocr_transcripts.json để Server tra cứu tức thì (< 0.0001s) khi thi đấu.

Cách chạy:
  .\\venv\\Scripts\\python.exe scripts/batch_ocr_extract.py --limit_videos 5
  .\\venv\\Scripts\\python.exe scripts/batch_ocr_extract.py (chạy toàn bộ)
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Các nơi lưu file JSON kết quả
OUTPUT_JSON_PATHS = [
    Path("D:/uploads/ocr_transcripts.json"),
    PROJECT_ROOT / "uploads" / "ocr_transcripts.json",
]


def find_keyframe_folders():
    """Tìm tất cả các thư mục chứa keyframe video trên D: và C:"""
    candidate_roots = [
        Path("D:/uploads"),
        Path("D:/keyframes"),
        PROJECT_ROOT / "uploads",
        PROJECT_ROOT / "data",
    ]
    video_dirs = {}
    for root in candidate_roots:
        if not root.exists():
            continue
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            s_name = sub.name.lower()
            if "keyframe" in s_name or s_name.startswith("l"):
                kf_sub = sub / "keyframes" if (sub / "keyframes").is_dir() else sub
                for vid_dir in kf_sub.iterdir():
                    if vid_dir.is_dir() and vid_dir.name not in video_dirs:
                        video_dirs[vid_dir.name] = vid_dir
    return video_dirs


def run_batch_ocr(limit_videos: int = None, target_video: str = None):
    print("=" * 65)
    print("🔍 BẮT ĐẦU CHẠY BATCH OCR (PADDLEOCR) ĐỘC LẬP")
    print("=" * 65)

    video_folders = find_keyframe_folders()
    print(f"📁 Tìm thấy tổng cộng: {len(video_folders)} thư mục video keyframe.")

    if target_video:
        if target_video in video_folders:
            video_folders = {target_video: video_folders[target_video]}
        else:
            print(f"❌ Không tìm thấy video: {target_video}")
            return

    video_keys = sorted(list(video_folders.keys()))
    if limit_videos:
        video_keys = video_keys[:limit_videos]
        print(f"⚙️ Giới hạn chạy {limit_videos} video đầu tiên.")

    # 1. Nạp cache đã có từ trước (Resumable)
    ocr_database: Dict[str, Dict[str, str]] = {}
    primary_out = OUTPUT_JSON_PATHS[0] if OUTPUT_JSON_PATHS[0].parent.exists() else OUTPUT_JSON_PATHS[1]
    primary_out.parent.mkdir(parents=True, exist_ok=True)

    for op in OUTPUT_JSON_PATHS:
        if op.is_file() and op.stat().st_size > 5:
            try:
                with open(op, "r", encoding="utf-8") as f:
                    ocr_database = json.load(f)
                print(f"ℹ️ Đã nạp lại {len(ocr_database)} video đã quét từ trước.")
                break
            except Exception:
                pass

    # 2. Khởi tạo PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("⏳ Đang khởi tạo PaddleOCR v4 Tiếng Việt...")
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='vi')
        print("✅ PaddleOCR đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi nạp PaddleOCR: {e}")
        return

    started_all = time.perf_counter()
    total_frames_scanned = 0

    # 3. Lặp qua từng video và từng ảnh keyframe
    for v_idx, v_name in enumerate(video_keys, 1):
        v_path = video_folders[v_name]
        
        # Nếu video đã quét đủ frame, có thể bỏ qua
        if v_name in ocr_database and len(ocr_database[v_name]) > 0:
            print(f"[{v_idx}/{len(video_keys)}] ⏭️ Bỏ qua {v_name} (Đã có {len(ocr_database[v_name])} frames).")
            continue

        images = sorted(list(v_path.glob("*.jpg")) + list(v_path.glob("*.webp")) + list(v_path.glob("*.png")))
        if not images:
            continue

        print(f"[{v_idx}/{len(video_keys)}] ⏳ Đang quét {v_name} ({len(images)} frames)...")
        v_results = {}
        t_v0 = time.perf_counter()

        for img in images:
            f_stem = img.stem
            try:
                result = ocr_engine.ocr(str(img), cls=True)
                if result and result[0]:
                    lines = [line[1][0] for line in result[0] if line and line[1]]
                    text = " ".join(lines).strip()
                else:
                    text = ""
                if text:
                    v_results[f_stem] = text
                    total_frames_scanned += 1
            except Exception as img_err:
                pass

        ocr_database[v_name] = v_results
        elapsed_v = time.perf_counter() - t_v0
        print(f"   ✅ Xong {v_name} ({len(v_results)} frames có chữ, {elapsed_v:.1f}s)")

        # Lưu checkpoint sau mỗi video
        for op in OUTPUT_JSON_PATHS:
            try:
                op.parent.mkdir(parents=True, exist_ok=True)
                with open(op, "w", encoding="utf-8") as f:
                    json.dump(ocr_database, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    print("=" * 65)
    print(f"🎉 HOÀN TẤT BATCH OCR ({len(ocr_database)} video) TRONG {time.perf_counter() - started_all:.1f}s!")
    print(f"📁 File kết quả: {primary_out}")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chạy Batch OCR trước ngày thi")
    parser.add_argument("--limit_videos", type=int, default=None, help="Số lượng video tối đa muốn quét thử")
    parser.add_argument("--video", type=str, default=None, help="Chỉ định cụ thể 1 video ID (ví dụ: L21_V001)")
    args = parser.parse_args()

    run_batch_ocr(limit_videos=args.limit_videos, target_video=args.video)
