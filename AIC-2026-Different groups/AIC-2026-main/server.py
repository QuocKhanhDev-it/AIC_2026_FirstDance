import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config, UPLOAD_DIR
from database import init_db, get_faiss_store
from indexer import query_search_text
from services.trake_service import perform_trake_search
from services.vqa_service import interact_kisc
from services.ocr_service import process_image
from services.asr_service import get_transcript_for_time
from services.export_service import export_to_zip

# Khởi tạo DB
init_db()

app = FastAPI(title="AIC 2026 - Multimodal Retrieval Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
EXPORTS_DIR = PROJECT_ROOT / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MULTI-DRIVE SMART MEDIA RESOLVER (TỰ ĐỘNG DÒ TÌM ẢNH/VIDEO TRÊN MỌI Ổ ĐĨA D:, C:)
# ==============================================================================

# Bộ nhớ đệm ánh xạ: { "L21_V001": Path("D:/uploads/Keyframes_L21_extracted/keyframes/L21_V001") }
_KEYFRAME_FOLDER_CACHE: Dict[str, Path] = {}
_VIDEO_FILE_CACHE: Dict[str, Path] = {}
_AUDIO_FILE_CACHE: Dict[str, Path] = {}
_CACHE_INITIALIZED = False


def _build_media_caches():
    """Tự động quét và index vị trí các thư mục keyframe, video, audio cực nhanh"""
    global _CACHE_INITIALIZED
    if _CACHE_INITIALIZED:
        return

    t0 = time.perf_counter()
    print("[Media Resolver] Đang lập bản đồ thư mục media trên ổ đĩa...", flush=True)
    
    # 1. Quét các thư mục đích xác định trước
    candidate_roots = [
        Path("D:/uploads"),
        Path("D:/keyframes"),
        PROJECT_ROOT / "uploads",
        PROJECT_ROOT / "data",
        Path("D:/")
    ]

    for root in candidate_roots:
        if not root.exists():
            continue
        try:
            # Quét các thư mục con cấp 1 (e.g. Keyframes_L21_extracted, video audio, objects...)
            for sub in root.iterdir():
                if not sub.is_dir():
                    continue
                s_name = sub.name.lower()
                
                # A. Thư mục keyframe: Keyframes_L21_extracted
                if "keyframe" in s_name or s_name.startswith("l"):
                    kf_sub = sub / "keyframes" if (sub / "keyframes").is_dir() else sub
                    for vid_dir in kf_sub.iterdir():
                        if vid_dir.is_dir() and vid_dir.name not in _KEYFRAME_FOLDER_CACHE:
                            _KEYFRAME_FOLDER_CACHE[vid_dir.name] = vid_dir

                # B. Thư mục audio/video (bao gồm cả các thư mục lồng nhau như videos L21-L30/Videos_L21_a)
                if "video" in s_name or "audio" in s_name:
                    for ext in ['*.mp4', '*.mkv', '*.avi', '*.webm']:
                        for vf in sub.rglob(ext):
                            if vf.stem not in _VIDEO_FILE_CACHE:
                                _VIDEO_FILE_CACHE[vf.stem] = vf
                    for ext in ['*.mp3', '*.wav', '*.m4a', '*.aac']:
                        for af in sub.rglob(ext):
                            if af.stem not in _AUDIO_FILE_CACHE:
                                _AUDIO_FILE_CACHE[af.stem] = af

        except Exception as e:
            print(f"[Media Resolver] Lưu ý khi quét {root}: {e}", flush=True)

    _CACHE_INITIALIZED = True
    print(f"✅ [Media Resolver] Đã lập bản đồ xong trong {time.perf_counter() - t0:.2f}s: {len(_KEYFRAME_FOLDER_CACHE)} thư mục keyframe, {len(_VIDEO_FILE_CACHE)} video MP4, {len(_AUDIO_FILE_CACHE)} audio MP3.", flush=True)


def find_image_file(rel_path: str) -> Optional[Path]:
    """
    Phân giải đường dẫn ảnh keyframe siêu tốc (< 0.0001s):
    - Tự động map từ cache thư mục trên D: hoặc C:
    - Hỗ trợ fallback .jpg <-> .webp <-> .png
    - Hỗ trợ số thứ tự frame: 001.jpg, 1.jpg, 079.jpg, 79.jpg
    """
    _build_media_caches()
    clean_path = rel_path.replace("\\", "/").strip("/")
    path_obj = Path(clean_path)
    
    parts = clean_path.split("/")
    if len(parts) >= 2:
        video_id = parts[-2]
        frame_name = parts[-1]
    else:
        video_id = path_obj.stem.split("_")[0] if "_" in path_obj.stem else path_obj.stem
        frame_name = path_obj.name

    frame_stem = Path(frame_name).stem
    
    # 1. Tìm thông qua Cache thư mục Video ID
    folder_path = _KEYFRAME_FOLDER_CACHE.get(video_id)
    if folder_path and folder_path.is_dir():
        try:
            ordinal_num = int(frame_stem)
            candidate_names = [
                f"{ordinal_num:03d}.jpg", f"{ordinal_num}.jpg",
                f"{ordinal_num:03d}.webp", f"{ordinal_num}.webp",
                f"{ordinal_num:03d}.png", f"{ordinal_num}.png",
                f"{video_id}_{ordinal_num:03d}.jpg", f"{video_id}_{ordinal_num}.jpg"
            ]
        except ValueError:
            candidate_names = [frame_name, f"{frame_stem}.jpg", f"{frame_stem}.webp", f"{frame_stem}.png"]

        for c_name in candidate_names:
            img_p = folder_path / c_name
            if img_p.is_file():
                return img_p

    # 2. Fallback quét trên các root nếu chưa có trong cache
    for root in [Path("D:/uploads"), Path("D:/keyframes"), PROJECT_ROOT / "uploads"]:
        if not root.exists():
            continue
        candidate_paths = [
            root / f"Keyframes_{video_id[:3]}_extracted" / "keyframes" / video_id / f"{frame_stem}.jpg",
            root / f"Keyframes_{video_id[:3]}_extracted" / "keyframes" / video_id / f"{int(frame_stem) if frame_stem.isdigit() else 0:03d}.jpg",
            root / "keyframes" / video_id / f"{frame_stem}.jpg",
            root / video_id / f"{frame_stem}.jpg",
            root / rel_path
        ]
        for cp in candidate_paths:
            if cp.is_file():
                _KEYFRAME_FOLDER_CACHE[video_id] = cp.parent
                return cp

    return None


def find_video_file(video_filename: str) -> Optional[Path]:
    """Tìm kiếm file video trong cache hoặc các ổ đĩa"""
    _build_media_caches()
    v_stem = Path(video_filename.strip()).stem
    
    # 1. Tra trong cache video
    if v_stem in _VIDEO_FILE_CACHE:
        v_path = _VIDEO_FILE_CACHE[v_stem]
        if v_path.is_file():
            return v_path
            
    # 2. Quét dự phòng
    for root in [Path("D:/uploads"), PROJECT_ROOT / "uploads", PROJECT_ROOT / "data"]:
        if not root.exists():
            continue
        for ext in [".mp4", ".mkv", ".avi", ".webm"]:
            candidate = root / "Videos_L21_a_extracted" / "video" / f"{v_stem}{ext}"
            if candidate.is_file():
                _VIDEO_FILE_CACHE[v_stem] = candidate
                return candidate
            candidate2 = root / f"{v_stem}{ext}"
            if candidate2.is_file():
                _VIDEO_FILE_CACHE[v_stem] = candidate2
                return candidate2

    return None


def find_audio_file(audio_filename: str) -> Optional[Path]:
    """Tìm kiếm file audio trong cache hoặc các ổ đĩa"""
    _build_media_caches()
    a_stem = Path(audio_filename.strip()).stem
    
    if a_stem in _AUDIO_FILE_CACHE:
        a_path = _AUDIO_FILE_CACHE[a_stem]
        if a_path.is_file():
            return a_path

    for root in [Path("D:/uploads"), PROJECT_ROOT / "uploads"]:
        if not root.exists():
            continue
        for ext in [".mp3", ".wav", ".m4a"]:
            candidate = root / "video audio" / f"{a_stem}{ext}"
            if candidate.is_file():
                _AUDIO_FILE_CACHE[a_stem] = candidate
                return candidate
            candidate2 = root / "video" / f"{a_stem}{ext}"
            if candidate2.is_file():
                _AUDIO_FILE_CACHE[a_stem] = candidate2
                return candidate2

    return None


# ==========================================
# CÁC ENDPOINT SERVE FILE TĨNH & MEDIA
# ==========================================

@app.get("/images/{image_path:path}")
def serve_image(image_path: str):
    """Phục vụ ảnh keyframe (.jpg, .webp, .png) từ mọi thư mục trên D: và C:"""
    file_path = find_image_file(image_path)
    if file_path and file_path.is_file():
        media_type = "image/webp" if file_path.suffix.lower() == ".webp" else "image/jpeg"
        return FileResponse(str(file_path), media_type=media_type)
    return HTMLResponse(content=f"<h3>Image not found: {image_path}</h3>", status_code=404)


@app.get("/videos/{video_name:path}")
def serve_video(video_name: str):
    """Phục vụ phát video MP4"""
    video_path = find_video_file(video_name)
    if video_path and video_path.is_file():
        return FileResponse(str(video_path), media_type="video/mp4")
    return HTMLResponse(content=f"<h3>Video not found: {video_name}</h3>", status_code=404)


@app.get("/audios/{audio_name:path}")
def serve_audio(audio_name: str):
    """Phục vụ phát audio MP3 dự phòng khi chưa có video MP4"""
    audio_path = find_audio_file(audio_name)
    if audio_path and audio_path.is_file():
        return FileResponse(str(audio_path), media_type="audio/mpeg")
    return HTMLResponse(content=f"<h3>Audio not found: {audio_name}</h3>", status_code=404)


# ==========================================
# SCHEMAS & API ENDPOINTS
# ==========================================

class KisRequest(BaseModel):
    query: str
    limit: Optional[int] = 60

class TrakeRequest(BaseModel):
    events: List[str]
    limit: Optional[int] = 30

class VqaRequest(BaseModel):
    query: str

class CartItem(BaseModel):
    type: str
    video: str
    frame: str
    frames: Optional[List[str]] = []
    answer: Optional[str] = ""

class ExportRequest(BaseModel):
    cart: List[CartItem]


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>Không tìm thấy file frontend/index.html</h2>", status_code=404)


@app.post("/api/search/kis")
def search_kis(req: KisRequest):
    if not req.query.strip():
        return {"results": []}
    try:
        limit_target = req.limit or 60
        fetch_limit = max(300, limit_target * 5)
        raw_results = query_search_text(req.query.strip(), filetype_filter="Tat ca", limit=fetch_limit)
        
        results = []
        seen_scenes = {}
        
        for r in raw_results:
            if len(results) >= limit_target:
                break
                
            video_id = r.get("video_id") or "Unknown"
            frame_idx = str(r.get("frame_idx", "0"))
            pts_time = float(r.get("pts_time", 0.0))
            ordinal = r.get("ordinal", 1)
            similarity = float(r.get("similarity", 0.0))
            
            # Lọc khử trùng lặp các frame sát nhau trong cùng 1 video (cách nhau <= 2.0s)
            if video_id != "Unknown" and video_id in seen_scenes:
                is_dup = any(abs(seen_t - pts_time) <= 2.0 for seen_t in seen_scenes[video_id])
                if is_dup:
                    continue
            else:
                seen_scenes[video_id] = []
            seen_scenes[video_id].append(pts_time)
            
            img_rel_path = f"{video_id}/{int(ordinal):03d}.jpg"
            has_video = find_video_file(video_id) is not None
            has_audio = find_audio_file(video_id) is not None
            
            results.append({
                "video": video_id,
                "frame": frame_idx,
                "ordinal": ordinal,
                "time": round(pts_time, 2),
                "path": img_rel_path,
                "similarity": round(similarity, 4),
                "score": round(similarity * 100, 1),
                "has_video": has_video,
                "has_audio": has_audio
            })
            
        return {"results": results, "total": len(results)}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.post("/api/search/trake")
def search_trake(req: TrakeRequest):
    valid_events = [e.strip() for e in req.events if e.strip()]
    if len(valid_events) < 2:
        return {"results": [], "message": "Cần ít nhất 2 sự kiện"}
    try:
        sequences = perform_trake_search(valid_events, limit=req.limit or 30)
        formatted_results = []
        for seq in sequences:
            if not seq:
                continue
            first_item = seq[0]
            video_id = first_item.get("video_id", "N/A")
            avg_sim = sum(float(item.get("similarity", 0.0)) for item in seq) / len(seq)
            
            seq_frames = []
            for item in seq:
                ordinal = item.get("ordinal", 1)
                seq_frames.append({
                    "video": item.get("video_id", video_id),
                    "frame": str(item.get("frame_idx", "0")),
                    "time": round(float(item.get("pts_time", 0.0)), 2),
                    "path": f"{item.get('video_id', video_id)}/{int(ordinal):03d}.jpg",
                    "similarity": round(float(item.get("similarity", 0.0)), 4)
                })
                
            formatted_results.append({
                "video": video_id,
                "frame": seq_frames[0]["frame"],
                "time": seq_frames[0]["time"],
                "path": seq_frames[0]["path"],
                "score": round(avg_sim * 100, 1),
                "full_sequence": seq_frames
            })
            
        return {"results": formatted_results}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.post("/api/search/vqa")
def search_vqa(req: VqaRequest):
    if not req.query.strip():
        return {"answer": "", "candidates": []}
    try:
        vqa_res = interact_kisc(req.query.strip(), filetype_filter="Tat ca")
        candidates = vqa_res.get("candidates", [])
        formatted_cands = []
        for c in candidates:
            vid = c.get("video_id", "")
            ordinal = c.get("ordinal", 1)
            formatted_cands.append({
                "video": vid,
                "frame": str(c.get("frame_idx", "")),
                "time": round(float(c.get("pts_time", 0.0)), 2),
                "path": f"{vid}/{int(ordinal):03d}.jpg",
                "similarity": round(float(c.get("similarity", 0.0)), 4)
            })
        return {
            "answer": vqa_res.get("answer", "Không tìm thấy câu trả lời."),
            "results": formatted_cands
        }
    except Exception as e:
        return {"answer": f"Lỗi VQA: {e}", "results": []}


@app.get("/api/details")
def get_details(video: str = Query(...), frame: str = Query(...), image_path: str = Query(...), pts_time: float = Query(0.0)):
    # 1. OCR
    try:
        real_img = find_image_file(image_path)
        if real_img and real_img.is_file():
            ocr_data = process_image(str(real_img))
            ocr_result = ocr_data.get("extracted_text", "") or "Không có văn bản trong ảnh."
        else:
            ocr_result = "Chưa tìm thấy file ảnh để quét OCR."
    except Exception as e:
        ocr_result = f"Lỗi OCR: {e}"

    # 2. ASR
    try:
        asr_result = get_transcript_for_time(video, pts_time=pts_time)
    except Exception as e:
        asr_result = f"Lỗi ASR: {e}"

    # 3. Media Availability
    has_video = find_video_file(video) is not None
    has_audio = find_audio_file(video) is not None

    return {
        "ocr": ocr_result,
        "asr": asr_result,
        "has_video": has_video,
        "has_audio": has_audio
    }


@app.post("/api/export")
def export_submission(req: ExportRequest):
    try:
        cart_dicts = [item.dict() for item in req.cart]
        zip_file_path = export_to_zip(cart_dicts, str(EXPORTS_DIR))
        return FileResponse(zip_file_path, filename=os.path.basename(zip_file_path), media_type="application/zip")
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print("🔥 Đang khởi động AIC 2026 Multimodal Retrieval Server...")
    print(f"👉 Thư mục dự án: {PROJECT_ROOT}")
    
    # Nạp trước FAISS Store và Media Cache
    get_faiss_store()
    _build_media_caches()
    
    # Lắng nghe trên 0.0.0.0 để cho phép tất cả thành viên trong mạng LAN/Wi-Fi truy cập
    uvicorn.run(app, host="0.0.0.0", port=8000)
