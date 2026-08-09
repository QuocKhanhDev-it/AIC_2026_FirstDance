from __future__ import annotations

import argparse
from pathlib import Path

from .config import BENCH_ROOT
from .safe_report import generate_report
from .runners import asr, ocr


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR then ASR and generate the final report")
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / "results")
    args = parser.parse_args()
    common = ["--output-dir", str(args.output_dir)]
    if args.include_unreviewed:
        common.append("--include-unreviewed")
    ocr.run(ocr.build_parser().parse_args(common))
    asr.run(asr.build_parser().parse_args(common))
    print(generate_report(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
