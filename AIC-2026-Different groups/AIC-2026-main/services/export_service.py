"""
Export Service - Đóng gói kết quả nộp bài thi AIC 2026 thành file ZIP chuẩn
"""
import os
import csv
import json
import zipfile
import time
from pathlib import Path
from typing import List, Dict, Any

def export_to_zip(cart_items: List[Dict[str, Any]], export_dir: str) -> str:
    """
    Nhận danh sách các mục đã chọn trong giỏ hàng (Cart) và đóng gói thành file ZIP nộp bài.
    """
    os.makedirs(export_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    zip_filename = f"aic2026_submission_{timestamp_str}.zip"
    zip_path = os.path.join(export_dir, zip_filename)
    
    csv_path = os.path.join(export_dir, "submission.csv")
    json_path = os.path.join(export_dir, "submission.json")
    
    # 1. Ghi file CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Type", "Video_ID", "Frame_ID", "Sequence_Frames", "Answer_Text"])
        for idx, item in enumerate(cart_items, 1):
            q_type = item.get("type", "kis")
            video = item.get("video", "")
            frame = item.get("frame", "")
            frames_seq = ",".join(str(x) for x in item.get("frames", [])) if item.get("frames") else frame
            answer = item.get("answer", "")
            writer.writerow([idx, q_type, video, frame, frames_seq, answer])

    # 2. Ghi file JSON
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_submissions": len(cart_items),
            "items": cart_items
        }, f, ensure_ascii=False, indent=2)

    # 3. Đóng gói vào file ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
        zf.write(json_path, arcname="submission.json")

    # Xóa file tạm
    try:
        os.remove(csv_path)
        os.remove(json_path)
    except Exception:
        pass

    return zip_path
