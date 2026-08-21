"""
asr_processor.py — Module xử lý nhận dạng giọng nói ASR từ Video (PhoWhisper-small Production)

Hỗ trợ:
1. Đọc video/file âm thanh (.mp4, .m4a, .wav).
2. Tự động nhận diện GPU CUDA (RTX 2060) với compute_type float16 / int8 để đạt tốc độ tối đa.
3. Chạy PhoWhisper-small với return_timestamps=True và VAD filter.
4. Ánh xạ các phân đoạn (start_time, end_time, text) tới keyframe tương ứng trong master.parquet theo pts_time.
5. Xuất dữ liệu ra asr.parquet với cơ chế lưu Checkpoint nguyên tử (Atomic Write).
"""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pipeline_OCR_ASR.config import (
    ASR_MODEL_PATH,
    ASR_DEVICE,
    ASR_COMPUTE_TYPE,
    ASR_PARQUET_PATH,
    DATA_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Tắt các log nội bộ từ CTranslate2 và Faster-Whisper để giữ thanh tiến độ hiển thị sạch đẹp
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("ctranslate2").setLevel(logging.WARNING)


def format_time(seconds: float) -> str:
    """Chuyển đổi số giây thành định dạng hh:mm:ss hoặc mm:ss."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


class VideoASRProcessor:
    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 compute_type: Optional[str] = None,
                 beam_size: int = 5):
        self.model_path = model_path or ASR_MODEL_PATH
        self.device = device or ASR_DEVICE
        self.compute_type = compute_type or ASR_COMPUTE_TYPE
        self.beam_size = beam_size
        self.model = None

    def _init_pipeline(self, cpu_threads: int = 4):
        """Khởi tạo mô hình Faster-Whisper CT2 nếu chưa được nạp."""
        if self.model is None:
            logging.info("Đang nạp mô hình ASR %s bằng CTranslate2 (faster-whisper) trên %s (compute_type: %s, beam_size: %d)...",
                         self.model_path, self.device, self.compute_type, self.beam_size)
            from faster_whisper import WhisperModel

            try:
                self.model = WhisperModel(
                    self.model_path,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=cpu_threads
                )
            except Exception as e:
                logging.warning("Không thể khởi tạo ASR trên %s (%s). Thử fallback về CPU int8...", self.device, e)
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_path,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads
                )

    def transcribe_video(self, video_path: Path) -> List[Dict[str, Any]]:
        """Trích xuất lời nói + timestamp từ 1 file video."""
        self._init_pipeline()
        if not video_path.exists():
            logging.warning("File video không tồn tại: %s", video_path)
            return []

        try:
            # Chạy nhận diện với bộ lọc VAD để loại bỏ khoảng lặng, tăng tốc độ
            segments, info = self.model.transcribe(
                str(video_path),
                language="vi",
                beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            results = []
            for s in segments:
                text = str(s.text).strip()
                if text:
                    results.append({
                        "start_time": float(s.start),
                        "end_time": float(s.end),
                        "asr_text": text
                    })
            return results
        except Exception as e:
            logging.error("Lỗi khi transcribe video %s: %s", video_path, e)
            return []

    def map_segments_to_keyframes(self, df_vid_master: pd.DataFrame, segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ánh xạ văn bản ASR theo (start_time, end_time) vào từng row_id keyframe theo pts_time."""
        rows = []
        for r in df_vid_master.itertuples():
            pts = float(r.pts_time)
            matched_texts = []
            for s in segs:
                if (s["start_time"] - 1.0) <= pts <= (s["end_time"] + 1.0):
                    matched_texts.append(s["asr_text"])

            asr_combined = " ".join(matched_texts) if matched_texts else ""
            rows.append({
                "row_id": int(r.row_id),
                "video_id": str(r.video_id),
                "kf_n": int(r.kf_n),
                "asr_text": asr_combined
            })
        return rows

    def process_dataset(self, df_master: pd.DataFrame, base_dir: Optional[Path] = None,
                        output_path: Path = ASR_PARQUET_PATH, max_workers: int = 1) -> pd.DataFrame:
        """Chạy ASR với tự động lưu Checkpoint nguyên tử và giao diện tiến độ trực quan."""
        import multiprocessing as mp

        output_path.parent.mkdir(parents=True, exist_ok=True)
        search_dir = base_dir if base_dir and base_dir.exists() else DATA_DIR

        # 1. Nạp Checkpoint (Chỉ tính video đã hoàn thành thực sự nếu có text hoặc đã ghi nhận)
        existing_vids = set()
        existing_rows: List[Dict[str, Any]] = []

        if output_path.exists() and output_path.stat().st_size > 0:
            try:
                df_existing = pd.read_parquet(output_path)
                if not df_existing.empty and "video_id" in df_existing.columns:
                    non_empty_count = (df_existing["asr_text"].astype(str).str.strip() != "").sum() if "asr_text" in df_existing.columns else 0
                    if non_empty_count > 0:
                        existing_rows = df_existing.to_dict("records")
                        existing_vids = set(df_existing["video_id"].unique())
                        logging.info("📌 TỰ ĐỘNG NẠP CHECKPOINT: Đã có %d video trong file ASR (%d dòng có text).",
                                     len(existing_vids), non_empty_count)
                    else:
                        logging.warning("⚠️ File checkpoint cũ toàn bộ rỗng (0 text). Sẽ reset checkpoint để chạy lại chuẩn xác!")
            except Exception as e:
                logging.warning("Không thể nạp checkpoint: %s", e)

        unique_vids = list(df_master["video_id"].unique())
        vids_to_process = [v for v in unique_vids if v not in existing_vids]

        if not vids_to_process:
            logging.info("✅ Tất cả video (%d) đã được ASR. Không cần chạy thêm.", len(unique_vids))
            return pd.DataFrame(existing_rows)

        logging.info("Đang quét tìm %d video cần chạy tại %s...", len(vids_to_process), search_dir)
        video_paths_map = {}
        for root, dirs, files in os.walk(search_dir):
            if ".zip" in root or "__MACOSX" in root:
                continue
            for f in files:
                if f.lower().endswith((".mp4", ".mkv", ".avi", ".m4a", ".wav")):
                    vid_stem = Path(f).stem
                    video_paths_map[vid_stem] = Path(root) / f

        tasks = []
        for vid in vids_to_process:
            p = video_paths_map.get(vid)
            if p:
                tasks.append((vid, p))

        total_tasks = len(tasks)
        logging.info("Tìm thấy %d / %d video trên đĩa. Bắt đầu xử lý (Device: %s, Compute: %s, Beam-size: %d)...",
                     total_tasks, len(vids_to_process), self.device, self.compute_type, self.beam_size)

        if total_tasks == 0:
            logging.warning("Không tìm thấy file video nào tại %s để chạy ASR!", search_dir)
            return pd.DataFrame(existing_rows)

        t_start = time.time()
        processed_count = 0

        # Nếu dùng CUDA GPU: Chạy 1 process để tối ưu hóa nhân Tensor Core và VRAM GPU
        effective_workers = 1 if self.device == "cuda" else max(1, min(4, max_workers))

        if effective_workers == 1:
            self._init_pipeline(cpu_threads=4)
            for vid, vid_path in tasks:
                segs = self.transcribe_video(vid_path)
                df_vid_master = df_master[df_master["video_id"] == vid]
                vid_rows = self.map_segments_to_keyframes(df_vid_master, segs)
                existing_rows.extend(vid_rows)

                processed_count += 1
                elapsed = time.time() - t_start
                curr_speed = processed_count / elapsed if elapsed > 0 else 0.0
                eta_sec = (total_tasks - processed_count) / curr_speed if curr_speed > 0 else 0.0
                pct = (processed_count / total_tasks) * 100

                print(f"\r[ASR Progress] 🎙️ {processed_count:,}/{total_tasks:,} ({pct:.2f}%) | "
                      f"Speed: {curr_speed*60:.1f} vid/min | Elapsed: {format_time(elapsed)} | "
                      f"ETA: {format_time(eta_sec)}", end="", flush=True)

                if processed_count % 5 == 0 or processed_count == total_tasks:
                    print()
                    df_current = pd.DataFrame(existing_rows).drop_duplicates(subset=["row_id"], keep="last").sort_values("row_id")
                    tmp_file = output_path.with_suffix(".tmp.parquet")
                    df_current.to_parquet(tmp_file, index=False)
                    tmp_file.replace(output_path)
                    logging.info("💾 Checkpoint đã lưu an toàn! [%d / %d videos | %d keyframes]",
                                 processed_count, total_tasks, len(df_current))
            print()
        else:
            total_cores = mp.cpu_count()
            threads_per_worker = max(1, total_cores // effective_workers)
            ctx = mp.get_context('spawn')

            with ctx.Pool(
                processes=effective_workers,
                initializer=init_asr_worker,
                initargs=(self.model_path, self.device, self.compute_type, threads_per_worker, self.beam_size)
            ) as pool:
                for vid, segs, err in pool.imap_unordered(run_asr_worker, tasks):
                    if err:
                        logging.error("Lỗi ASR video %s: %s", vid, err)
                        continue

                    df_vid_master = df_master[df_master["video_id"] == vid]
                    vid_rows = self.map_segments_to_keyframes(df_vid_master, segs)
                    existing_rows.extend(vid_rows)

                    processed_count += 1
                    elapsed = time.time() - t_start
                    curr_speed = processed_count / elapsed if elapsed > 0 else 0.0
                    eta_sec = (total_tasks - processed_count) / curr_speed if curr_speed > 0 else 0.0
                    pct = (processed_count / total_tasks) * 100

                    print(f"\r[ASR Progress] 🎙️ {processed_count:,}/{total_tasks:,} ({pct:.2f}%) | "
                          f"Speed: {curr_speed*60:.1f} vid/min | Elapsed: {format_time(elapsed)} | "
                          f"ETA: {format_time(eta_sec)}", end="", flush=True)

                    if processed_count % 5 == 0 or processed_count == total_tasks:
                        print()
                        df_current = pd.DataFrame(existing_rows).drop_duplicates(subset=["row_id"], keep="last").sort_values("row_id")
                        tmp_file = output_path.with_suffix(".tmp.parquet")
                        df_current.to_parquet(tmp_file, index=False)
                        tmp_file.replace(output_path)
                        logging.info("💾 Checkpoint đã lưu an toàn! [%d / %d videos | %d keyframes]",
                                     processed_count, total_tasks, len(df_current))
            print()

        total_time = time.time() - t_start
        df_final = pd.DataFrame(existing_rows).drop_duplicates(subset=["row_id"], keep="last").sort_values("row_id")
        logging.info("🎉 HOÀN TẤT TOÀN BỘ ASR! Tổng số dòng: %d | Thời gian: %s",
                     len(df_final), format_time(total_time))
        return df_final


def init_asr_worker(model_path: str, device: str, compute_type: str, threads: int, beam_size: int = 5):
    global _worker_asr_proc
    _worker_asr_proc = VideoASRProcessor(model_path=model_path, device=device, compute_type=compute_type, beam_size=beam_size)
    _worker_asr_proc._init_pipeline(cpu_threads=threads)


def run_asr_worker(args: Tuple[str, Path]) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    vid, vid_path = args
    global _worker_asr_proc
    try:
        segs = _worker_asr_proc.transcribe_video(vid_path)
        return vid, segs, None
    except Exception as e:
        return vid, [], str(e)
