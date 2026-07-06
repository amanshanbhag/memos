"""Realistic serving benchmarks (Phase 2).

Wraps vLLM's own OpenAI server (`vllm serve`) and open-loop load generator
(`vllm bench serve`) to measure latency-SLO curves under an arrival process --
the regime that steady-state batch sweeps cannot reach because vLLM's V1
scheduler admission-controls the running batch.
"""

from memos.serving.bench import parse_vllm_bench_json, run_bench_serve
from memos.serving.server import (
    PrometheusPoller,
    VLLMServer,
    parse_prometheus_metrics,
)

__all__ = [
    "VLLMServer",
    "PrometheusPoller",
    "run_bench_serve",
    "parse_vllm_bench_json",
    "parse_prometheus_metrics",
]
