#!/usr/bin/env bash
set -Eeuo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
CACHE_ROOT="${AIC_AI_CACHE_ROOT:-D:/Library/ai_cache}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
PADDLE_INDEX_URL="${PADDLE_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"

DO_PERSIST=0
DO_INSTALL=0
DO_DOWNLOAD=0
DO_VERIFY=0

usage() {
  cat <<'EOF'
Usage: bash set_up.sh [--persist] [--install] [--download-models] [--verify]
                      [--cache-root D:/path]

Without options, runs all steps. It always configures cache variables for the
current process. It never creates a virtual environment or installs globally.
EOF
}

if [[ $# -eq 0 ]]; then
  DO_PERSIST=1; DO_INSTALL=1; DO_DOWNLOAD=1; DO_VERIFY=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --persist) DO_PERSIST=1 ;;
    --install) DO_INSTALL=1 ;;
    --download-models) DO_DOWNLOAD=1 ;;
    --verify) DO_VERIFY=1 ;;
    --cache-root)
      [[ $# -ge 2 ]] || { echo "--cache-root requires a value" >&2; exit 2; }
      CACHE_ROOT="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Required existing venv not found: $VENV_PYTHON" >&2
  exit 1
fi

export AIC_AI_CACHE_ROOT="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export EASYOCR_MODULE_PATH="$CACHE_ROOT/easyocr"
export PADDLE_PDX_CACHE_HOME="$CACHE_ROOT/paddle"
export AIC_VIETOCR_CACHE="$CACHE_ROOT/vietocr"
export AIC_WHISPER_CACHE="$CACHE_ROOT/whisper"

mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$EASYOCR_MODULE_PATH" \
  "$PADDLE_PDX_CACHE_HOME" "$AIC_VIETOCR_CACHE" "$AIC_WHISPER_CACHE"

if [[ $DO_PERSIST -eq 1 ]]; then
  powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File "$BENCH_ROOT/scripts/configure_cache.ps1" -CacheRoot "$CACHE_ROOT"
fi

if [[ $DO_INSTALL -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
  "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
  "$VENV_PYTHON" -m pip install "paddlepaddle-gpu==3.2.0" --index-url "$PADDLE_INDEX_URL"
  "$VENV_PYTHON" -m pip install -r "$BENCH_ROOT/requirements-benchmark.txt"
  "$VENV_PYTHON" -m pip freeze > "$BENCH_ROOT/requirements-lock.txt"
fi

if [[ $DO_DOWNLOAD -eq 1 ]]; then
  "$VENV_PYTHON" "$BENCH_ROOT/scripts/prefetch_models.py" --config "$BENCH_ROOT/configs/models.yaml"
fi

if [[ $DO_VERIFY -eq 1 ]]; then
  "$VENV_PYTHON" "$BENCH_ROOT/scripts/verify_environment.py"
fi

echo "Setup completed. Python: $VENV_PYTHON"
echo "Model cache root: $CACHE_ROOT"
