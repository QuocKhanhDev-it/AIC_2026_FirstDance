#!/bin/bash
echo 'This file is deprecated. Run ../set_up.sh from BenchmarkOCRASR.' >&2
exit 1
# ==============================================================================
# setup_environment.sh
# Setup môi trường + tải model cho benchmark pipeline AIC2026
# Yêu cầu: GPU (khuyến nghị >=16GB VRAM để chạy song song vài model),
#          CUDA 11.8+ hoặc 12.x, Python 3.10
# Chạy: bash setup_environment.sh
# ==============================================================================

set -e  # dừng ngay nếu có lỗi

# ------------------------------------------------------------------------------
# [BƯỚC 0] Chuyển cache model (HuggingFace/PaddleOCR/Whisper) sang ổ D
# Xem chi tiết: move_cache_to_drive_d.md
# Chỉ ép phần checkpoint/cache model — KHÔNG đụng tới venv hay thư mục code,
# những cái đó tự quyết định vị trí khi tạo. Tự bỏ qua nếu không chạy trên
# Windows hoặc máy không có ổ D.
# ------------------------------------------------------------------------------
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || -n "$WINDIR" ]] && [ -d "/d" ]; then
    echo "=== [0/7] Set biến môi trường trỏ cache model sang D:\\Library\\ai_cache ==="
    mkdir -p "/d/Library/ai_cache/huggingface"
    mkdir -p "/d/Library/ai_cache/torch"
    mkdir -p "/d/Library/ai_cache/paddle"
    mkdir -p "/d/Library/ai_cache/whisper"

    export HF_HOME="D:\Library\ai_cache\huggingface"
    export HUGGINGFACE_HUB_CACHE="D:\Library\ai_cache\huggingface\hub"
    export TRANSFORMERS_CACHE="D:\Library\ai_cache\huggingface\transformers"
    export TORCH_HOME="D:\Library\ai_cache\torch"

    echo "    HF_HOME, TRANSFORMERS_CACHE, TORCH_HOME đã trỏ sang D:\\Library\\ai_cache"
    echo "    (Nên set các biến này CỐ ĐỊNH qua Environment Variables của Windows"
    echo "     để không phải export lại mỗi lần mở terminal mới — xem mục 2,"
    echo "     Cách A trong move_cache_to_drive_d.md)"
else
    echo "=== [0/7] Không phải Windows hoặc không thấy ổ D — bỏ qua bước set cache path ==="
fi

ENV_NAME="aic2026-bench"
PYTHON_VERSION="3.10"

echo "=== [1/7] Tạo virtual environment ==="
# Dùng conda nếu có, không thì dùng venv
if command -v conda &> /dev/null; then
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    source activate $ENV_NAME
else
    python3 -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi

echo "=== [2/7] Cài PyTorch (chỉnh lại URL theo phiên bản CUDA của máy bạn) ==="
# Kiểm tra CUDA version bằng: nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/7] Cài thư viện core ==="
pip install \
    numpy pandas scipy scikit-learn \
    opencv-python-headless pillow \
    tqdm pyyaml requests \
    faiss-gpu \
    rank_bm25 elasticsearch \
    matplotlib seaborn \
    jiwer editdistance \
    psutil gputil nvidia-ml-py3

echo "=== [4/7] Cài model OCR ==="
# PaddleOCR + PaddlePaddle GPU
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
pip install paddleocr

# VietOCR
pip install vietocr

# EasyOCR (baseline so sánh)
pip install easyocr

echo "=== [5/7] Cài model ASR ==="
# PhoWhisper (dùng qua HuggingFace transformers)
pip install transformers accelerate sentencepiece

# OpenAI Whisper (baseline so sánh)
pip install openai-whisper

# ffmpeg cần cho xử lý audio - kiểm tra đã có chưa
if ! command -v ffmpeg &> /dev/null; then
    echo "!!! Chưa có ffmpeg, cài bằng: sudo apt install ffmpeg (Ubuntu/Debian)"
fi

echo "=== [6/7] Cài model Visual-Text (CLIP family) + VLM ==="
pip install open_clip_torch
pip install git+https://github.com/openai/CLIP.git

# SigLIP, AltCLIP, Jina-CLIP, Vintern, Qwen2-VL đều load qua transformers/HF hub
pip install transformers>=4.40 timm einops

# Qwen2-VL cần thêm
pip install qwen-vl-utils

# Dịch thuật (nếu dùng chiến lược dịch query Việt->Anh)
pip install sentencepiece sacremoses

echo ""
echo "=== Cài xong thư viện. Bắt đầu tải checkpoint model (chạy python) ==="

python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

os.makedirs("checkpoints", exist_ok=True)

MODELS = {
    # --- CLIP family ---
    "clip-vit-b32":  "openai/clip-vit-base-patch32",       # baseline BTC
    "clip-vit-l14":  "openai/clip-vit-large-patch14",
    "siglip2":       "google/siglip2-base-patch16-multilingual",
    "altclip":       "BAAI/AltCLIP",
    "jina-clip-v2":  "jinaai/jina-clip-v2",

    # --- ASR ---
    "phowhisper-small":  "vinai/PhoWhisper-small",
    "phowhisper-medium": "vinai/PhoWhisper-medium",
    "phowhisper-large":  "vinai/PhoWhisper-large",

    # --- VLM ---
    "vintern-1b":    "5CD-AI/Vintern-1B-v3_5",
    "qwen2-vl-2b":   "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b":   "Qwen/Qwen2-VL-7B-Instruct",  # bỏ dòng này nếu VRAM hạn chế

    # --- Dịch thuật (tuỳ chọn) ---
    "envit5":        "VietAI/envit5-translation",
}

for local_name, repo_id in MODELS.items():
    dest = f"checkpoints/{local_name}"
    print(f"--- Đang tải {repo_id} -> {dest} ---")
    try:
        snapshot_download(repo_id=repo_id, local_dir=dest, local_dir_use_symlinks=False)
    except Exception as e:
        print(f"!!! Lỗi khi tải {repo_id}: {e}")
        print("    -> Có thể model cần huggingface-cli login, hoặc tên repo đã đổi.")

print("\n=== Hoàn tất tải checkpoint (kiểm tra log lỗi phía trên nếu có) ===")
EOF

echo ""
echo "=== [Tuỳ chọn] PaddleOCR/VietOCR/Whisper tự tải weight lần đầu chạy inference ==="
echo "    (không cần snapshot_download riêng, sẽ cache vào ~/.paddleocr, ~/.cache, v.v.)"

echo ""
echo "=== SETUP HOÀN TẤT ==="
echo "Kích hoạt môi trường bằng:"
echo "  conda activate $ENV_NAME   (nếu dùng conda)"
echo "  source $ENV_NAME/bin/activate   (nếu dùng venv)"
echo ""
echo "Lưu ý:"
echo "  - Qwen2-VL-7B và một số model lớn cần >=16GB VRAM, cân nhắc bỏ nếu máy yếu."
echo "  - faiss-gpu có thể cần build lại nếu không match CUDA version, có thể đổi sang faiss-cpu."
echo "  - Một số checkpoint HuggingFace là gated model, cần: huggingface-cli login"
