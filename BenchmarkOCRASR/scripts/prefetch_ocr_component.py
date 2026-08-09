from __future__ import annotations

import argparse
import os
from pathlib import Path


def d_path(name: str) -> Path:
    path = Path(os.environ[name]).resolve()
    if path.drive.lower() != 'd:':
        raise RuntimeError('{} must point to drive D'.format(name))
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    import requests
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=['paddle', 'easyocr', 'vietocr'])
    component = parser.parse_args().component
    if component == 'paddle':
        d_path('PADDLE_PDX_CACHE_HOME')
        import paddle
        import sys
        import types
        modelscope_stub = types.ModuleType('modelscope')
        modelscope_stub.snapshot_download = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError('ModelScope is disabled; use the configured Paddle model source')
        )
        sys.modules['modelscope'] = modelscope_stub
        from paddleocr import PaddleOCR
        PaddleOCR(
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='latin_PP-OCRv5_mobile_rec',
            device='cpu',
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    elif component == 'easyocr':
        root = d_path('EASYOCR_MODULE_PATH')
        import easyocr
        easyocr.Reader(
            ['vi', 'en'], gpu=False,
            model_storage_directory=str(root / 'model'),
            user_network_directory=str(root / 'user_network'),
        )
    else:
        root = d_path('AIC_VIETOCR_CACHE')
        files = {
            root / 'configs/base.yml': 'https://vocr.vn/data/vietocr/config/base.yml',
            root / 'configs/vgg-transformer.yml': 'https://vocr.vn/data/vietocr/config/vgg-transformer.yml',
            root / 'weights/vgg_transformer.pth': 'https://vocr.vn/data/vietocr/vgg_transformer.pth',
        }
        for destination, url in files.items():
            download(url, destination)
    print('Prefetched {}'.format(component))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
