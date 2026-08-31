"""
ASR Service - Hệ thống nhận diện lời thoại âm thanh (Audio Speech Recognition):
  - 100% Cache-First: Tra cứu siêu tốc (< 0.0001s) từ file asr_transcripts.json đã trích xuất trước.
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

# Danh sách nơi tìm file ASR JSON đã trích xuất sẵn
TRANSCRIPT_JSON_CANDIDATES = [
    Path("D:/uploads/asr_transcripts.json"),
    PROJECT_ROOT / "uploads" / "asr_transcripts.json",
    PROJECT_ROOT / "asr_transcripts.json",
]

_cached_transcripts: Optional[Dict[str, Any]] = None


def _load_transcripts_json() -> Dict[str, Any]:
    """Tải và lưu vào bộ nhớ RAM toàn bộ dữ liệu lời thoại từ JSON đã tiền xử lý"""
    global _cached_transcripts
    if _cached_transcripts is not None:
        return _cached_transcripts

    for p in TRANSCRIPT_JSON_CANDIDATES:
        if p.is_file() and p.stat().st_size > 5:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data:
                    _cached_transcripts = data
                    print(f"✅ [ASR] Đã nạp thành công cache lời thoại từ {p.name} ({len(data)} video).", flush=True)
                    return _cached_transcripts
            except Exception as e:
                print(f"⚠️ [ASR] Lỗi đọc {p}: {e}", flush=True)

    _cached_transcripts = {}
    return _cached_transcripts


def get_transcript_for_time(video_id: str, pts_time: float = 0.0, window_sec: float = 15.0) -> str:
    """
    Tra cứu lời thoại video tại thời điểm pts_time (giây) trong < 0.0001s:
    Args:
        video_id: Tên video (ví dụ: "L21_V001")
        pts_time: Mốc thời gian của keyframe (giây)
        window_sec: Khoảng thời gian mở rộng tìm phụ đề xung quanh (mặc định 15s)
    """
    transcripts = _load_transcripts_json()
    v_stem = Path(video_id.strip()).stem
    
    candidates_keys = [v_stem, f"{v_stem}.mp3", f"{v_stem}.mp4", video_id.strip()]
    entry = None
    for c in candidates_keys:
        if c in transcripts:
            entry = transcripts[c]
            break

    if entry is not None:
        if isinstance(entry, str):
            return entry

        # Nếu entry có dạng { "full_text": "...", "segments": [...] }
        segments = entry.get("segments", [])
        if segments:
            matched_texts = []
            for seg in segments:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                # Khớp đoạn nằm quanh pts_time ± window_sec
                if (start <= pts_time <= end) or (abs(start - pts_time) <= window_sec) or (abs(end - pts_time) <= window_sec):
                    ts_str = f"[{int(start//60):02d}:{int(start%60):02d}]"
                    matched_texts.append(f"{ts_str} {seg.get('text', '').strip()}")

            if matched_texts:
                return "\n".join(matched_texts[:5])

        if entry.get("full_text"):
            return entry["full_text"][:300] + ("..." if len(entry["full_text"]) > 300 else "")

    return "ℹ️ Không tìm thấy lời thoại (hoặc chưa chạy batch ASR cho video này)."
