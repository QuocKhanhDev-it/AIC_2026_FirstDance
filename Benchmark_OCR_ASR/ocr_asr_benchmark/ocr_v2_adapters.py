from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .config import require_cache_on_d
from .utils.geometry import polygon_to_xyxy


@dataclass(frozen=True)
class Recognition:
    text: str
    confidence: float | None

    def to_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


class Recognizer(Protocol):
    def recognize_line(self, image: Image.Image) -> Recognition: ...


def _easy_reader(params: dict[str, Any]) -> Any:
    root = require_cache_on_d('EASYOCR_MODULE_PATH')
    import easyocr
    return easyocr.Reader(
        params.get('languages', ['vi', 'en']),
        gpu=params.get('gpu', True),
        model_storage_directory=str(root / 'model'),
        user_network_directory=str(root / 'user_network'),
    )


class EasyOCRRecognizer:
    '''Recognition-only EasyOCR adapter; Reader.detect is never called.'''

    def __init__(self, params: dict[str, Any]):
        self.reader = _easy_reader(params)

    def recognize_line(self, image: Image.Image) -> Recognition:
        outputs = self.reader.recognize(
            np.asarray(image.convert('L')), horizontal_list=None, free_list=None,
            detail=1, paragraph=False,
        )
        if not outputs:
            return Recognition('', None)
        texts = [str(item[1]).strip() for item in outputs if str(item[1]).strip()]
        scores = [float(item[2]) for item in outputs if len(item) > 2]
        return Recognition(' '.join(texts), min(scores) if scores else None)


class VietOCRRecognizer:
    def __init__(self, params: dict[str, Any]):
        cache_root = require_cache_on_d('AIC_VIETOCR_CACHE')
        import yaml
        config_name = params.get('vietocr_config', 'vgg_transformer')
        filenames = {
            'vgg_transformer': 'vgg-transformer.yml',
            'resnet_transformer': 'resnet_transformer.yml',
            'resnet_fpn_transformer': 'resnet_fpn_transformer.yml',
        }
        base_path = cache_root / 'configs' / 'base.yml'
        model_path = cache_root / 'configs' / filenames[config_name]
        if not base_path.exists() or not model_path.exists():
            raise RuntimeError('VietOCR config is not prefetched; run set_up.sh first')
        config = yaml.safe_load(base_path.read_text(encoding='utf-8'))
        config.update(yaml.safe_load(model_path.read_text(encoding='utf-8')))
        weights_url = str(config['weights'])
        weights_path = cache_root / 'weights' / Path(weights_url).name
        if not weights_path.exists():
            raise RuntimeError('VietOCR weights are not prefetched; run set_up.sh first')
        config['weights'] = str(weights_path)
        config['device'] = params.get('device', 'cuda')
        config['predictor']['beamsearch'] = False
        from vietocr.tool.predictor import Predictor
        self.predictor = Predictor(config)

    def recognize_line(self, image: Image.Image) -> Recognition:
        text, probability = self.predictor.predict(image.convert('RGB'), return_prob=True)
        confidence = None if probability is None else float(probability)
        return Recognition(str(text).strip(), confidence)


def _paddle_data(result: Any) -> dict[str, Any]:
    value = result.json if hasattr(result, 'json') else result
    value = value() if callable(value) else value
    if isinstance(value, str):
        value = json.loads(value)
    return value.get('res', value) if isinstance(value, dict) else {}


class PaddleRecognizer:
    def __init__(self, params: dict[str, Any]):
        require_cache_on_d('PADDLE_PDX_CACHE_HOME')
        from paddleocr import TextRecognition
        self.model = TextRecognition(
            model_name=params.get('model_name', 'latin_PP-OCRv5_mobile_rec'),
            device=params.get('device', 'gpu'),
        )

    def recognize_line(self, image: Image.Image) -> Recognition:
        outputs = list(self.model.predict(np.asarray(image.convert('RGB'))))
        if not outputs:
            return Recognition('', None)
        data = _paddle_data(outputs[0])
        text = data.get('rec_text', data.get('text', ''))
        score = data.get('rec_score', data.get('score'))
        return Recognition(str(text).strip(), float(score) if score is not None else None)


class EasyOCRDetector:
    '''The frozen detector used only by the ROI/gate benchmark.'''

    def __init__(self, params: dict[str, Any]):
        self.reader = _easy_reader(params)

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        horizontal_groups, free_groups = self.reader.detect(np.asarray(image.convert('RGB')))
        horizontal = horizontal_groups[0] if horizontal_groups else []
        free = free_groups[0] if free_groups else []
        polygons = [
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            for x1, x2, y1, y2 in horizontal
        ]
        polygons.extend(free)
        return [{
            'polygon': [[float(x), float(y)] for x, y in polygon],
            'bbox_xyxy': polygon_to_xyxy(polygon),
        } for polygon in polygons]


def load_recognizer(config: dict[str, Any]) -> Recognizer:
    family = config['family']
    if family == 'easyocr_recognizer':
        return EasyOCRRecognizer(config.get('params', {}))
    if family == 'vietocr_recognizer':
        return VietOCRRecognizer(config.get('params', {}))
    if family == 'paddle_recognizer':
        return PaddleRecognizer(config.get('params', {}))
    raise ValueError(f'Unknown OCR v2 recognizer family: {family}')
