from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from memos.types import BenchmarkResult, InferenceConfig, MetricSample


def save_result(result: BenchmarkResult, path: str | Path) -> Path:
    """Save a BenchmarkResult to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    return path


def load_result(path: str | Path) -> BenchmarkResult:
    """Load a BenchmarkResult from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    raw["metrics"] = [MetricSample(**m) for m in raw.get("metrics", [])]
    inf = raw.get("inference_config")
    if isinstance(inf, dict):
        raw["inference_config"] = InferenceConfig(**inf)
    return BenchmarkResult(**raw)
