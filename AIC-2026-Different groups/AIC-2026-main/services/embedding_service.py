"""
Embedding Service - Đã đồng bộ chuẩn Không gian Vector SigLIP 1152 Chiều (1152D):
  - Image & Text Model: SigLIP (ViT-SO400M-14-SigLIP-384, pretrained='webli')
  - Hỗ trợ tối ưu hóa suy luận CPU (torch.inference_mode)
  - Text Search: Nhập từ khóa tiếng Anh trực tiếp qua SigLIP tokenizer
"""
import sys
import torch
import open_clip
import numpy as np
from typing import List
from PIL import Image

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

# ─── THIẾT BỊ VÀ CẤU HÌNH ───────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-SO400M-14-SigLIP-384"
PRETRAINED = "webli"
VECTOR_DIM = 1152

# Tối ưu hóa số luồng xử lý trên CPU
if DEVICE == "cpu":
    try:
        torch.set_num_threads(4)
    except Exception:
        pass

# ─── BỘ NHỚ ĐỆM MÔ HÌNH (CACHE) ─────────────────────────────────────────────
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip():
    """Tải mô hình SigLIP ViT-SO400M-14-SigLIP-384 chuẩn 1152D"""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        print(f"[Embedding] Đang tải mô hình SigLIP {MODEL_NAME} ({VECTOR_DIM}D) trên thiết bị: {DEVICE}...", flush=True)
        try:
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
            )
            _clip_tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            _clip_model = _clip_model.to(DEVICE).eval()
            print(f"✅ [Embedding] Hệ thống SigLIP ({VECTOR_DIM}D) đã sẵn sàng trên {DEVICE}!", flush=True)
        except Exception as e:
            print(f"❌ [Embedding] Lỗi tải SigLIP: {e}", flush=True)
            raise e

    return _clip_model, _clip_preprocess, _clip_tokenizer


# ─── CÁC HÀM API CHÍNH DÙNG CHO INDEXER & SEARCH ─────────────────────────

def get_image_embedding(image_path: str) -> List[float]:
    """Tạo Vector đại diện cho ảnh (1152D) bằng SigLIP"""
    try:
        model, preprocess, _ = _load_clip()
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().numpy().tolist()
    except Exception as e:
        print(f"❌ [Embedding] Lỗi tạo vector hình ảnh {image_path}: {e}", flush=True)
        return []


def get_image_embeddings_batch(image_paths: List[str]) -> List[List[float]]:
    """Tạo Vector đại diện cho một lô ảnh (1152D)"""
    if not image_paths:
        return []
    try:
        model, preprocess, _ = _load_clip()
        imgs = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                img = preprocess(Image.open(path).convert("RGB"))
                imgs.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"⚠️ [Embedding] Lỗi đọc ảnh {path}: {e}", flush=True)

        if not imgs:
            return [[] for _ in image_paths]

        batch_tensor = torch.stack(imgs).to(DEVICE)
        with torch.inference_mode():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        features_list = features.cpu().numpy().tolist()
        final_results = [[] for _ in image_paths]
        for idx, feat in zip(valid_indices, features_list):
            final_results[idx] = feat

        return final_results
    except Exception as e:
        print(f"❌ [Embedding] Lỗi tạo vector lô hình ảnh: {e}", flush=True)
        return [[] for _ in image_paths]


def get_clip_text_embedding(query: str) -> List[float]:
    """
    Tạo Vector 1152D từ câu truy vấn tiếng Anh bằng SigLIP Text Encoder.
    """
    if not query or not query.strip():
        return []
    try:
        model, _, tokenizer = _load_clip()
        text_tokens = tokenizer([query.strip()]).to(DEVICE)
        with torch.inference_mode():
            features = model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"❌ [Embedding Error]: {e}", flush=True)
        return []


def get_text_embedding(text: str) -> List[float]:
    """Alias cho get_clip_text_embedding"""
    return get_clip_text_embedding(text)


def get_embedding(text: str) -> List[float]:
    """Hàm tương thích ngược"""
    return get_clip_text_embedding(text)