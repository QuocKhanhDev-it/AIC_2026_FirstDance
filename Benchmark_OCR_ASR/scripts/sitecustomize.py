"""Make the benchmark package importable when executing scripts directly."""
from pathlib import Path
import sys

benchmark_root = Path(__file__).resolve().parents[1]
if str(benchmark_root) not in sys.path:
    sys.path.insert(0, str(benchmark_root))
