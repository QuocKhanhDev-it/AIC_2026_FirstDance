"""
asr_processor.py — Module xử lý nhận dạng giọng nói ASR từ Video (PhoWhisper-small Production)

Hỗ trợ:
1. Đọc video/file âm thanh (.mp4, .m4a, .wav).
2. Chạy PhoWhisper-small với return_timestamps=True.
3. Ánh xạ các phân đoạn (start_time, end_time, text) tới keyframe tương ứng trong master.parquet theo pts_time.
4. Xuất dữ liệu ra asr.parquet và asr_by_keyframe.parquet.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from pipeline_OCR_ASR.config import ASR_MODEL_ID, ASR_PARQUET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VideoASRProcessor:
    def __init__(self, model_id: str = ASR_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

    def _init_pipeline(self):
        if self.pipeline is None:
            logging.info("Đang nạp mô hình ASR %s trên thiết bị %s...", self.model_id, self.device)
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
            
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained(self.model_id)
            
            self.pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=dtype,
                device=0 if self.device == "cuda" else -1,
                chunk_length_s=30,
                batch_size=1,
            )

    def transcribe_video(self, video_path: Path) -> List[Dict[str, Any]]:
        """Trích xuất lời nói + timestamp từ 1 file video."""
        self._init_pipeline()
        if not video_path.exists():
            logging.warning("File video không tồn tại: %s", video_path)
            return []

        try:
            res = self.pipeline(
                str(video_path),
                return_timestamps=True,
                generate_kwargs={"language": "vi", "task": "transcribe", "num_beams": 5, "do_sample": False}
            )
            chunks = res.get("chunks", [])
            segments = []
            for c in chunks:
                ts = c.get("timestamp", (0.0, 0.0))
                start_t = float(ts[0]) if ts[0] is not None else 0.0
                end_t = float(ts[1]) if ts[1] is not None else start_t + 2.0
                text = str(c.get("text", "")).strip()
                if text:
                    segments.append({
                        "start_time": start_t,
                        "end_time": end_t,
                        "asr_text": text
                    })
            return segments
        except Exception as e:
            logging.error("Lỗi khi transcribe video %s: %s", video_path, e)
            return []

    def map_segments_to_keyframes(self, df_master: pd.DataFrame, video_segments: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Ánh xạ văn bản ASR theo (start_time, end_time) vào từng row_id keyframe theo pts_time."""
        rows = []
        for row_id, r in df_master.iterrows():
            vid = str(r["video_id"])
            pts = float(r["pts_time"])
            segs = video_segments.get(vid, [])
            
            # Lấy tất cả phân đoạn ASR mà pts_time nằm trong khoảng [start - 1.0, end + 1.0] (dung sai thời gian)
            matched_texts = []
            for s in segs:
                if (s["start_time"] - 1.0) <= pts <= (s["end_time"] + 1.0):
                    matched_texts.append(s["asr_text"])
            
            asr_combined = " ".join(matched_texts) if matched_texts else ""
            rows.append({
                "row_id": int(r["row_id"]),
                "video_id": vid,
                "kf_n": int(r["kf_n"]),
                "asr_text": asr_combined
            })
            
        return pd.DataFrame(rows)
