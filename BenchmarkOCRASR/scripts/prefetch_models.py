from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def drive_d_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    path = Path(value).resolve()
    if path.drive.lower() != "d:":
        raise RuntimeError(f"{name} must be on drive D, got {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    drive_d_path("HF_HOME")
    hf_hub = drive_d_path("HF_HUB_CACHE")
    easy_root = drive_d_path("EASYOCR_MODULE_PATH")
    drive_d_path("PADDLE_PDX_CACHE_HOME")
    drive_d_path("AIC_VIETOCR_CACHE")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    from huggingface_hub import snapshot_download
    whisper_root = drive_d_path('AIC_WHISPER_CACHE')
    for model in config['asr']:
        if model.get('enabled', True):
            local_dir = whisper_root / model['repo_id'].replace('/', '--')
            print('Downloading {} to {}'.format(model['repo_id'], local_dir))
            snapshot_download(
                model['repo_id'], local_dir=str(local_dir), cache_dir=str(hf_hub),
            )
    config['asr'] = []
    import subprocess
    import sys
    component_script = Path(__file__).with_name('prefetch_ocr_component.py')
    for component in ('paddle', 'easyocr', 'vietocr'):
        subprocess.run([sys.executable, str(component_script), component], check=True)
    return 0
    for model in config["asr"]:
        if model.get("enabled", True):
            print(f"Downloading {model['repo_id']}")
            snapshot_download(model["repo_id"], cache_dir=str(hf_hub))

    import easyocr
    print("Initializing EasyOCR weights")
    easyocr.Reader(
        ["vi", "en"], gpu=False,
        model_storage_directory=str(easy_root / "model"),
        user_network_directory=str(easy_root / "user_network"),
    )

    from paddleocr import PaddleOCR
    print("Initializing PaddleOCR PP-OCRv5 weights")
    PaddleOCR(
        lang="vi", ocr_version="PP-OCRv5", device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print("VietOCR checkpoint is resolved by its loader on first smoke run.")
    print('Initializing VietOCR checkpoint')
    from ocr_asr_benchmark.model_loader import PaddleVietOCRAdapter
    PaddleVietOCRAdapter({'vietocr_config': 'vgg_transformer', 'device': 'cpu'})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
