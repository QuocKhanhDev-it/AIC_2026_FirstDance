"""
Script chạy Batch Offline ASR (Whisper) độc lập trước ngày thi:
  - Quét toàn bộ file MP3 trong D:\uploads\video audio và C:\
  - Tự động nạp ffmpeg
  - Bóc tách lời thoại kèm timestamp theo từng video
  - Xuất ra file asr_transcripts.json để Server tra cứu tức thì (< 0.0001s) khi thi đấu.

Cách chạy:
  .\\venv\\Scripts\\python.exe scripts/batch_asr_transcribe.py --model tiny --max 5
  .\\venv\\Scripts\\python.exe scripts/batch_asr_transcribe.py --model base (chạy toàn bộ)
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

# Tự động cấu hình ffmpeg
def _setup_ffmpeg():
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = str(Path(ffmpeg_exe).parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        
        ffmpeg_plain = Path(ffmpeg_dir) / "ffmpeg.exe"
        if not ffmpeg_plain.exists() and Path(ffmpeg_exe).exists():
            import shutil
            try:
                shutil.copy2(ffmpeg_exe, ffmpeg_plain)
            except Exception:
                pass
    except Exception:
        pass

_setup_ffmpeg()

# Các nơi tìm file audio
AUDIO_SEARCH_DIRS = [
    Path("D:/uploads/video audio"),
    Path("D:/uploads/video"),
    Path("D:/uploads"),
    PROJECT_ROOT / "uploads" / "video audio",
    PROJECT_ROOT / "uploads" / "video",
    PROJECT_ROOT / "uploads",
]

# Các nơi lưu file JSON kết quả
OUTPUT_JSON_PATHS = [
    Path("D:/uploads/asr_transcripts.json"),
    PROJECT_ROOT / "uploads" / "asr_transcripts.json",
]


def find_all_audio_files():
    """Tìm tất cả các file âm thanh trên các ổ đĩa"""
    audio_files = {}
    for d in AUDIO_SEARCH_DIRS:
        if not d.exists():
            continue
        for ext in ["*.mp3", "*.wav", "*.m4a", "*.aac"]:
            for f in d.glob(ext):
                if f.stem not in audio_files:
                    audio_files[f.stem] = f
    return audio_files


def run_batch_asr(max_files: int = None, model_name: str = "base"):
    print("=" * 65)
    print(f"🎙️ BẮT ĐẦU CHẠY BATCH ASR (WHISPER '{model_name.upper()}') ĐỘC LẬP")
    print("=" * 65)

    audio_map = find_all_audio_files()
    print(f"📁 Tìm thấy tổng cộng: {len(audio_map)} file audio MP3/WAV.")

    if not audio_map:
        print("❌ Không tìm thấy file audio nào trong D:\\uploads\\video audio!")
        return

    audio_keys = sorted(list(audio_map.keys()))
    if max_files:
        audio_keys = audio_keys[:max_files]
        print(f"⚙️ Giới hạn chạy {max_files} file đầu tiên.")

    # 1. Nạp cache đã có từ trước (Resumable)
    transcripts: Dict[str, Any] = {}
    primary_out = OUTPUT_JSON_PATHS[0] if OUTPUT_JSON_PATHS[0].parent.exists() else OUTPUT_JSON_PATHS[1]
    primary_out.parent.mkdir(parents=True, exist_ok=True)

    for op in OUTPUT_JSON_PATHS:
        if op.is_file() and op.stat().st_size > 5:
            try:
                with open(op, "r", encoding="utf-8") as f:
                    transcripts = json.load(f)
                print(f"ℹ️ Đã nạp lại {len(transcripts)} file đã hoàn thành từ trước.")
                break
            except Exception:
                pass

    # 2. Khởi tạo Whisper
    try:
        import whisper
        print(f"⏳ Đang nạp mô hình Whisper '{model_name}' trên CPU...")
        model = whisper.load_model(model_name, device="cpu")
        print("✅ Mô hình Whisper đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi nạp Whisper: {e}")
        return

    started_all = time.perf_counter()
    count_new = 0

    # 3. Lặp qua từng file audio
    for idx, stem in enumerate(audio_keys, 1):
        af = audio_map[stem]
        if stem in transcripts:
            print(f"[{idx}/{len(audio_keys)}] ⏭️ Bỏ qua {af.name} (Đã có sẵn)")
            continue

        t0 = time.perf_counter()
        print(f"[{idx}/{len(audio_keys)}] ⏳ Đang xử lý: {af.name}...")
        try:
            res = model.transcribe(str(af), language="vi", fp16=False, verbose=False)
            segments = [
                {
                    "start": round(seg.get("start", 0.0), 2),
                    "end": round(seg.get("end", 0.0), 2),
                    "text": seg.get("text", "").strip()
                }
                for seg in res.get("segments", [])
            ]
            transcripts[stem] = {
                "filename": af.name,
                "full_text": res.get("text", "").strip(),
                "segments": segments
            }
            elapsed = time.perf_counter() - t0
            print(f"   ✅ Xong {af.name} ({len(segments)} đoạn, {elapsed:.1f}s)")
            count_new += 1

            # Lưu checkpoint mỗi video
            for op in OUTPUT_JSON_PATHS:
                try:
                    op.parent.mkdir(parents=True, exist_ok=True)
                    with open(op, "w", encoding="utf-8") as f:
                        json.dump(transcripts, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        except Exception as e:
            print(f"   ❌ Lỗi xử lý {af.name}: {e}")

    print("=" * 65)
    print(f"🎉 HOÀN TẤT TRÍCH XUẤT LỜI THOẠI ({len(transcripts)} files) TRONG {time.perf_counter() - started_all:.1f}s!")
    print(f"📁 File kết quả: {primary_out}")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chạy Batch ASR Whisper trước ngày thi")
    parser.add_argument("--max", type=int, default=None, help="Số file tối đa muốn chạy thử")
    parser.add_argument("--model", type=str, default="base", help="Tên whisper model (tiny, base, small)")
    args = parser.parse_args()

    run_batch_asr(max_files=args.max, model_name=args.model)
