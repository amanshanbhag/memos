"""Unit tests for the serving benchmark parsers (no GPU / server required).

Runnable directly (`python tests/test_serving_parsers.py`) or under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memos.serving.bench import parse_vllm_bench_json  # noqa: E402
from memos.serving.server import (  # noqa: E402
    _emit_engine_args,
    parse_prometheus_metrics,
)


def test_parse_vllm_bench_json_extracts_dynamic_percentiles() -> None:
    data = {
        "date": "20260706",
        "backend": "vllm",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_prompts": 500,
        "request_rate": 8.0,
        "duration": 62.5,
        "completed": 500,
        "request_throughput": 8.0,
        "output_throughput": 1024.0,
        "total_token_throughput": 9216.0,
        "mean_ttft_ms": 55.2,
        "median_ttft_ms": 50.0,
        "p90_ttft_ms": 70.0,
        "p99_ttft_ms": 120.0,
        "mean_tpot_ms": 12.5,
        "p99_tpot_ms": 25.0,
        "mean_itl_ms": 12.0,
        "p99_e2el_ms": 2500.0,
    }
    out = parse_vllm_bench_json(data)

    assert out["p99_ttft_ms"] == 120.0
    assert out["mean_tpot_ms"] == 12.5
    assert out["p99_e2el_ms"] == 2500.0
    assert out["output_throughput"] == 1024.0
    assert out["completed"] == 500.0
    # Non-metric / string fields are dropped.
    assert "backend" not in out
    assert "date" not in out
    assert "model_id" not in out


def test_parse_vllm_bench_json_ignores_bools_and_strings() -> None:
    data = {"p99_ttft_ms": "oops", "completed": True, "output_throughput": 100}
    out = parse_vllm_bench_json(data)
    assert "p99_ttft_ms" not in out  # string value rejected
    assert "completed" not in out  # bool rejected (bool is int subclass)
    assert out["output_throughput"] == 100.0


def test_parse_prometheus_metrics_strips_labels_last_wins() -> None:
    text = """
# HELP vllm:num_requests_running Number of running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama",engine="0"} 17.0
vllm:num_requests_waiting{model_name="llama"} 42
vllm:kv_cache_usage_perc{engine="0"} 1.0
vllm:num_preemptions_total{engine="0"} 5
# a trailing comment
vllm:gpu_cache_usage_perc 0.82
""".strip()
    out = parse_prometheus_metrics(text)

    assert out["vllm:num_requests_running"] == 17.0
    assert out["vllm:num_requests_waiting"] == 42.0
    assert out["vllm:kv_cache_usage_perc"] == 1.0
    assert out["vllm:num_preemptions_total"] == 5.0
    assert out["vllm:gpu_cache_usage_perc"] == 0.82


def test_parse_prometheus_metrics_defensive_on_garbage() -> None:
    text = 'not_a_metric\nbad{ line\nname{a="b"} notanumber\nok_metric 3.14'
    out = parse_prometheus_metrics(text)
    assert out == {"ok_metric": 3.14}


def test_emit_engine_args_bool_and_value_flags() -> None:
    argv = _emit_engine_args(
        {
            "kv_cache_dtype": "fp8",
            "gpu_memory_utilization": 0.5,
            "trust_remote_code": True,
            "enforce_eager": False,
            "max_model_len": 131072,
        }
    )
    assert "--kv-cache-dtype" in argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert "--gpu-memory-utilization" in argv
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.5"
    assert "--trust-remote-code" in argv  # True -> bare flag
    assert "--enforce-eager" not in argv  # False -> omitted
    assert "--max-model-len" in argv


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
