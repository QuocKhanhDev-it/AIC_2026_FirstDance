"""
VQA & KISC Service - Multimodal RAG cho AIC 2026:
  - Tìm kiếm KIS (FAISS) để tìm ảnh liên quan -> Gửi kèm ảnh + ngữ cảnh lên Gemini
  - Model: gemini-flash-latest (cập nhật tự động từ config)
  - Hỗ trợ Conversational KIS (nhiều vòng lặp)
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from google import genai
from google.genai.errors import ClientError
from google.genai import types as genai_types
from PIL import Image
from config import config
from indexer import query_search_text

try:
    from services.object_service import summarize_frame_objects
except Exception:
    summarize_frame_objects = None


def _safe_print(*values):
    msg = " ".join(str(v) for v in values)
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


_gemini_client_instance = None


def _get_gemini_client():
    global _gemini_client_instance
    if _gemini_client_instance is None:
        api_key = config.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            _gemini_client_instance = genai.Client(api_key=api_key)
    return _gemini_client_instance


def _resolve_image_path(cand: dict) -> str:
    """Tìm đường dẫn ảnh thực của kết quả FAISS candidate từ D: hoặc C:"""
    # Thử từ filepath trực tiếp
    fp = cand.get("filepath", "")
    if fp and os.path.exists(fp):
        return fp

    # Thử dùng find_image_file từ server module nếu khởi động qua server
    try:
        from server import find_image_file
        video_id = cand.get("video_id", "")
        ordinal = cand.get("ordinal", 1)
        rel_path = f"{video_id}/{int(ordinal):03d}.jpg"
        found = find_image_file(rel_path)
        if found and found.is_file():
            return str(found)
    except Exception:
        pass

    return ""


def interact_kisc(query_text: str, chat_history: list = None, filetype_filter: str = "Tat ca") -> dict:
    """
    VQA Multimodal RAG:
    1. Tìm 10-15 frame liên quan nhất từ FAISS (KIS)
    2. Gửi ảnh + metadata + câu hỏi lên Gemini
    3. Nhận câu trả lời thông minh
    """
    _safe_print(f"[VQA] Start; query='{query_text[:60]}'")
    client = _get_gemini_client()

    if not client:
        return {"answer": "Thieu GEMINI_API_KEY. Them vao file .env.", "candidates": []}

    # 1. Tim ung vien tu FAISS - dung "Tat ca" de lay tat ca image
    # filetype_filter map: "Hinh anh"/"Tat ca" -> "image" hoac tat ca
    effective_filter = "Tat ca"  # Lay tat ca image frame
    raw_candidates = query_search_text(query_text, filetype_filter=effective_filter, limit=15)
    if not raw_candidates:
        return {"answer": "Khong tim thay manh moi nao khop voi query.", "candidates": []}

    # 2. Xay dung Context
    context_lines = []
    valid_pil_images = []
    valid_image_bytes = []

    for i, cand in enumerate(raw_candidates[:10]):
        video_id = cand.get("video_id", "N/A")
        ordinal = cand.get("ordinal", 1)
        pts_time = cand.get("pts_time", 0.0)
        sim = cand.get("similarity", 0.0)

        line = f"[Frame {i+1}] Video={video_id} Frame={ordinal} Time={pts_time:.1f}s Score={sim:.3f}"
        context_lines.append(line)

        # Them anh (toi da 3 anh de khong qua tai token)
        if len(valid_pil_images) < 3:
            img_path = _resolve_image_path(cand)
            if img_path:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.thumbnail((768, 768))
                    valid_pil_images.append(img)
                    # Convert to bytes for genai
                    import io
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    valid_image_bytes.append(buf.getvalue())
                except Exception:
                    pass

    # 3. Tao Prompt
    context_text = "\n".join(context_lines)
    prompt = (
        f"Ban la AI ho tro tim kiem video AIC 2026.\n"
        f"Nguoi dung hoi: '{query_text}'\n\n"
        f"Danh sach cac frame lien quan tim duoc tu he thong:\n{context_text}\n\n"
        f"Dua vao cac anh keyframe duoc dinh kem va thong tin tren, hay phan tich va tra loi cau hoi cua nguoi dung. "
        f"Neu co the, hay chi ra frame nao (Video, Frame, Time) co kha nang cao nhat la dap an chinh xac."
    )

    # 4. Gui len Gemini
    model_name = config.gemini_model or "gemini-flash-latest"

    try:
        # Xay dung contents list: [image_bytes..., text_prompt]
        contents = []
        for img_bytes in valid_image_bytes:
            contents.append(
                genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )
        contents.append(prompt)

        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )
        _safe_print(f"[VQA] Thanh cong voi model={model_name}")
        return {"answer": response.text, "candidates": raw_candidates[:5]}

    except Exception as e:
        _safe_print(f"[VQA] LOI voi model={model_name}: {e}")
        # Fallback: chi gui text (khong anh)
        try:
            resp_fallback = client.models.generate_content(
                model=model_name,
                contents=[f"Query: {query_text}\nContext:\n{context_text}\nTra loi bang Tieng Viet ngan gon."]
            )
            return {"answer": resp_fallback.text, "candidates": raw_candidates[:5]}
        except Exception as e2:
            _safe_print(f"[VQA] Fallback cung loi: {e2}")
            return {
                "answer": f"VQA loi: {str(e2)[:200]}. Kiem tra GEMINI_API_KEY va model '{model_name}'.",
                "candidates": raw_candidates[:5]
            }