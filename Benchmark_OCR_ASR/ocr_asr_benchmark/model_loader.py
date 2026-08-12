from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import require_cache_on_d
from .utils.geometry import polygon_to_xyxy


def _paddle_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "json"):
        value = result.json
        value = value() if callable(value) else value
    elif hasattr(result, "to_dict"):
        value = result.to_dict()
    else:
        value = result
    if isinstance(value, str):
        import json
        value = json.loads(value)
    return value.get("res", value) if isinstance(value, dict) else {}


def _prepare_paddle_import() -> None:
    import sys
    import types
    import paddle
    if 'modelscope' not in sys.modules:
        modelscope_stub = types.ModuleType('modelscope')
        modelscope_stub.snapshot_download = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError('ModelScope is disabled; use the configured Paddle model source')
        )
        sys.modules['modelscope'] = modelscope_stub


class PaddleOCRAdapter:
    def __init__(self, params: dict[str, Any]):
        _prepare_paddle_import()
        require_cache_on_d("PADDLE_PDX_CACHE_HOME")
        from paddleocr import PaddleOCR
        self.model = PaddleOCR(**params)

    def predict(self, image_path: Path) -> list[dict[str, Any]]:
        outputs = list(self.model.predict(str(image_path)))
        predictions: list[dict[str, Any]] = []
        for output in outputs:
            data = _paddle_result(output)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [None] * len(texts))
            polygons = data.get("rec_polys") or data.get("dt_polys") or []
            for polygon, text, score in zip(polygons, texts, scores):
                predictions.append({
                    "bbox_xyxy": polygon_to_xyxy(np.asarray(polygon).tolist()),
                    "polygon": np.asarray(polygon).tolist(),
                    "text": str(text),
                    "confidence": float(score) if score is not None else None,
                })
        return predictions

    def recognize_crop(self, image: Image.Image) -> str:
        outputs = list(self.model.predict(np.asarray(image.convert("RGB"))))
        texts: list[str] = []
        for output in outputs:
            texts.extend(str(x) for x in _paddle_result(output).get("rec_texts", []))
        return " ".join(texts).strip()


class EasyOCRAdapter:
    def __init__(self, params: dict[str, Any]):
        root = require_cache_on_d("EASYOCR_MODULE_PATH")
        import easyocr
        self.model = easyocr.Reader(
            params.get("languages", ["vi", "en"]),
            gpu=params.get("gpu", True),
            model_storage_directory=str(root / "model"),
            user_network_directory=str(root / "user_network"),
        )

    def _convert(self, outputs: list[Any]) -> list[dict[str, Any]]:
        return [{
            "bbox_xyxy": polygon_to_xyxy(polygon),
            "polygon": polygon,
            "text": str(text),
            "confidence": float(confidence),
        } for polygon, text, confidence in outputs]

    def predict(self, image_path: Path) -> list[dict[str, Any]]:
        return self._convert(self.model.readtext(str(image_path), detail=1))

    def recognize_crop(self, image: Image.Image) -> str:
        outputs = self.model.readtext(np.asarray(image.convert("RGB")), detail=0, paragraph=True)
        return " ".join(map(str, outputs)).strip()


class PaddleVietOCRAdapter:
    '''Paddle detector plus VietOCR recognizer with every artifact cached on D.'''

    _CONFIG_NAMES = {
        'vgg_transformer': 'vgg-transformer.yml',
        'resnet_transformer': 'resnet_transformer.yml',
        'resnet_fpn_transformer': 'resnet_fpn_transformer.yml',
    }

    @staticmethod
    def _download(url: str, destination: Path) -> Path:
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        import requests
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + '.part')
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        partial.replace(destination)
        return destination

    def __init__(self, params: dict[str, Any]):
        detector_params = {
            'lang': 'vi', 'ocr_version': params.get('detector_ocr_version', 'PP-OCRv5'),
            'device': 'gpu' if params.get('device', 'cuda') == 'cuda' else 'cpu',
            'use_doc_orientation_classify': False, 'use_doc_unwarping': False,
            'use_textline_orientation': False,
        }
        self.detector = PaddleOCRAdapter(detector_params)
        cache_root = require_cache_on_d('AIC_VIETOCR_CACHE')
        import yaml
        config_name = params.get('vietocr_config', 'vgg_transformer')
        model_filename = self._CONFIG_NAMES[config_name]
        config_root = cache_root / 'configs'
        base_path = self._download(
            'https://vocr.vn/data/vietocr/config/base.yml', config_root / 'base.yml'
        )
        model_path = self._download(
            f'https://vocr.vn/data/vietocr/config/{model_filename}', config_root / model_filename
        )
        config = yaml.safe_load(base_path.read_text(encoding='utf-8'))
        config.update(yaml.safe_load(model_path.read_text(encoding='utf-8')))
        weights_url = str(config['weights'])
        weights_path = self._download(weights_url, cache_root / 'weights' / Path(weights_url).name)
        config['weights'] = str(weights_path)
        config['device'] = params.get('device', 'cuda')
        config['predictor']['beamsearch'] = False
        from vietocr.tool.predictor import Predictor
        self.recognizer = Predictor(config)

    def recognize_crop(self, image: Image.Image) -> str:
        return str(self.recognizer.predict(image.convert('RGB'))).strip()

    def predict(self, image_path: Path) -> list[dict[str, Any]]:
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        predictions = self.detector.predict(image_path)
        for prediction in predictions:
            x1, y1, x2, y2 = prediction['bbox_xyxy']
            crop = image.crop((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
            prediction['text'] = self.recognize_crop(crop)
        return predictions


class EasyVietOCRAdapter:
    '''EasyOCR detector plus VietOCR recognizer; both use one PyTorch runtime.'''

    def __init__(self, params: dict[str, Any]):
        self.detector = EasyOCRAdapter(params)
        cache_root = require_cache_on_d('AIC_VIETOCR_CACHE')
        import yaml
        config_name = params.get('vietocr_config', 'vgg_transformer')
        model_filename = PaddleVietOCRAdapter._CONFIG_NAMES[config_name]
        config_root = cache_root / 'configs'
        base_path = PaddleVietOCRAdapter._download(
            'https://vocr.vn/data/vietocr/config/base.yml', config_root / 'base.yml'
        )
        model_path = PaddleVietOCRAdapter._download(
            f'https://vocr.vn/data/vietocr/config/{model_filename}', config_root / model_filename
        )
        config = yaml.safe_load(base_path.read_text(encoding='utf-8'))
        config.update(yaml.safe_load(model_path.read_text(encoding='utf-8')))
        weights_url = str(config['weights'])
        weights_path = PaddleVietOCRAdapter._download(
            weights_url, cache_root / 'weights' / Path(weights_url).name
        )
        config['weights'] = str(weights_path)
        config['device'] = params.get('device', 'cuda')
        config['predictor']['beamsearch'] = False
        from vietocr.tool.predictor import Predictor
        self.recognizer = Predictor(config)

    def recognize_crop(self, image: Image.Image) -> str:
        return str(self.recognizer.predict(image.convert('RGB'))).strip()

    def predict(self, image_path: Path) -> list[dict[str, Any]]:
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        predictions = self.detector.predict(image_path)
        for prediction in predictions:
            x1, y1, x2, y2 = prediction['bbox_xyxy']
            crop = image.crop((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
            prediction['text'] = self.recognize_crop(crop)
        return predictions


def load_ocr_adapter(model_config: dict[str, Any]) -> Any:
    family = model_config['family']
    if family == 'paddleocr':
        return PaddleOCRAdapter(model_config['params'])
    if family == 'easyocr':
        return EasyOCRAdapter(model_config['params'])
    if family == 'easy_vietocr':
        return EasyVietOCRAdapter(model_config['params'])
    raise ValueError('Unknown OCR family: {}'.format(family))


class TransformersWhisperAdapter:
    '''Whisper adapter backed by a real local snapshot on D (no Windows symlinks).'''

    def __init__(self, model_config: dict[str, Any], generation: dict[str, Any]):
        cache_dir = require_cache_on_d('HF_HUB_CACHE')
        snapshot_root = require_cache_on_d('AIC_WHISPER_CACHE')
        repo_id = model_config['repo_id']
        local_dir = snapshot_root / repo_id.replace('/', '--')
        if not (local_dir / 'config.json').exists():
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id, local_dir=str(local_dir), cache_dir=str(cache_dir),
            )
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(local_dir), dtype=dtype, low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(str(local_dir))
        self.pipeline = pipeline(
            'automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor, dtype=dtype,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30, batch_size=1,
        )
        self.generation = generation

    def predict(self, audio_path: Path) -> dict[str, Any]:
        generate_kwargs = {
            'language': self.generation.get('language', 'vi'),
            'task': self.generation.get('task', 'transcribe'),
            'num_beams': self.generation.get('num_beams', 5),
            'do_sample': False,
        }
        return self.pipeline(
            str(audio_path),
            return_timestamps=self.generation.get('return_timestamps', True),
            generate_kwargs=generate_kwargs,
        )


def load_asr_adapter(model_config: dict[str, Any], generation: dict[str, Any]) -> Any:
    if model_config["family"] == "transformers_whisper":
        return TransformersWhisperAdapter(model_config, generation)
    raise ValueError(f"Unknown ASR family: {model_config['family']}")
