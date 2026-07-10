"""Open-loop serving benchmark orchestration around `vllm bench serve`.

`run_bench_serve` drives a live `VLLMServer` through a (request_rate x
max_concurrency) grid, invoking `vllm bench serve` for each point, parsing the
latency percentiles + throughput it reports, and attaching the pressure sample
collected from the server's /metrics during that point. The pure parser
helpers (`parse_vllm_bench_json`, `parse_prometheus_metrics`) are separated out
so they can be unit-tested without a GPU or a running server.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from memos.serving.server import (
    KVBMPoller,
    PrometheusPoller,
    VLLMServer,
    parse_prometheus_metrics,
)
from memos.types import BenchmarkResult, HardwareConfig, MetricSample

__all__ = [
    "run_bench_serve",
    "parse_vllm_bench_json",
    "parse_prometheus_metrics",
]

_LATENCY_RE = re.compile(r"^(mean|median|std|p\d+)_(ttft|tpot|itl|e2el)_ms$")

# vllm bench serve result keys we lift verbatim (throughput + accounting).
_THROUGHPUT_KEYS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "request_goodput",
    "completed",
    "duration",
    "total_input_tokens",
    "total_output_tokens",
    "num_prompts",
)


def parse_vllm_bench_json(data: dict[str, Any]) -> dict[str, float]:
    """Extract latency percentiles + throughput from a bench-serve result dict.

    Percentile keys are dynamic (they depend on --metric-percentiles), so we
    match any `{mean|median|std|pNN}_{ttft|tpot|itl|e2el}_ms` key rather than
    hard-coding a list. Non-numeric values are ignored.
    """
    out: dict[str, float] = {}
    for key, value in data.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if _LATENCY_RE.match(key) or key in _THROUGHPUT_KEYS:
            out[key] = float(value)
    return out


def _unit_for(name: str) -> str:
    if name.endswith("_ms"):
        return "ms"
    if name in ("request_throughput", "request_goodput"):
        return "req/s"
    if name in ("output_throughput", "total_token_throughput"):
        return "tokens/s"
    if name == "duration":
        return "s"
    if name in ("running_batch_peak", "waiting_peak"):
        return "requests"
    if name in ("kv_cache_usage_peak", "cpu_cache_usage_peak"):
        return "ratio"
    if name.startswith("kvbm"):
        if "hit_rate" in name:
            return "ratio"
        if "tokens" in name:
            return "tokens"
        if "blocks" in name:
            return "blocks"
        return "count"
    return "count"


def _run_one_bench(
    base_url: str,
    model: str,
    isl: int,
    osl: int,
    num_prompts: int,
    request_rate: str,
    max_concurrency: int | None,
    dataset: str,
    percentiles: str,
    result_dir: Path,
    tag: str,
    prefix_len: int = 0,
    range_ratio: str | None = None,
    dataset_path: str | None = None,
    num_prefixes: int = 0,
) -> dict[str, Any]:
    """Invoke `vllm bench serve` for one point and return its parsed result JSON.

    Workload realism knobs (random dataset): `prefix_len` prepends a fixed
    SHARED prefix to every request (`--random-prefix-len`), which drives KV
    reuse -- the total input length becomes `prefix_len + isl`. `range_ratio`
    (`--random-range-ratio`, in [0,1)) jitters ISL/OSL so lengths are
    heterogeneous instead of a single fixed value. `dataset_path` feeds
    non-random datasets (e.g. sharegpt) their trace file.

    For the `prefix_repetition` dataset, `num_prefixes` DISTINCT prefixes (each
    `prefix_len` tokens, `isl` suffix tokens, `osl` output tokens) are generated
    and each is repeated `num_prompts // num_prefixes` times. Unlike a single
    shared prefix (which dedups and RELIEVES pressure), a pool of distinct-but-
    reused prefixes larger than HBM both creates eviction pressure AND rewards
    reload-over-recompute -- the capacity regime where KV tiering can win.
    """
    result_file = f"{tag}.json"
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--base-url",
        base_url,
        "--dataset-name",
        dataset,
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(request_rate),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        percentiles,
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_file,
    ]
    if dataset == "random":
        cmd += [
            "--random-input-len",
            str(isl),
            "--random-output-len",
            str(osl),
        ]
        if prefix_len:
            cmd += ["--random-prefix-len", str(prefix_len)]
        if range_ratio is not None:
            cmd += ["--random-range-ratio", str(range_ratio)]
    elif dataset == "prefix_repetition":
        cmd += [
            "--prefix-repetition-prefix-len",
            str(prefix_len),
            "--prefix-repetition-suffix-len",
            str(isl),
            "--prefix-repetition-output-len",
            str(osl),
        ]
        if num_prefixes:
            cmd += ["--prefix-repetition-num-prefixes", str(num_prefixes)]
    if dataset_path:
        cmd += ["--dataset-path", dataset_path]
    if max_concurrency:
        cmd += ["--max-concurrency", str(max_concurrency)]

    subprocess.run(cmd, check=True)

    with open(result_dir / result_file) as f:
        return json.load(f)


def run_bench_serve(
    model: str,
    hw: HardwareConfig,
    tp: int,
    isl: int,
    osl: int,
    num_prompts: int,
    request_rates: list[str],
    max_concurrency: list[int] | None = None,
    engine_args: dict[str, Any] | None = None,
    dataset: str = "random",
    port: int = 8000,
    percentiles: str = "90,95,99",
    result_dir: str | None = None,
    server_log: str | None = None,
    env: dict[str, str] | None = None,
    warmup_prompts: int = 64,
    kvbm_metrics_port: int | None = None,
    prefix_len: int = 0,
    range_ratio: str | None = None,
    dataset_path: str | None = None,
    num_prefixes: int = 0,
) -> BenchmarkResult:
    """Sweep offered load against a live vLLM server and collect SLO + pressure.

    One BenchmarkResult with a MetricSample per (metric, grid-point). Each grid
    point is (request_rate x max_concurrency) at a fixed ISL/OSL; the arrival
    process is vLLM's Poisson generator. Pressure metrics come from the server's
    /metrics sampled during the point.

    A warmup pass (`warmup_prompts` requests at unthrottled rate, result
    discarded) runs first so the initial measured point is not contaminated by
    one-time torch.compile / CUDA-graph capture stalls (otherwise the lowest QPS
    point shows a huge TTFT tail as the first requests trigger compilation).

    When `kvbm_metrics_port` is set (KVBM KV-offload enabled), the KVBM metrics
    endpoint on that port is sampled alongside /metrics so each point also
    records tier-movement counters (offload/onboard blocks, cache hit rate) --
    the evidence that KV moved across tiers instead of being recomputed.

    Workload realism: `prefix_len` gives every request a shared fixed prefix
    (total input = prefix_len + isl) so KV blocks are reused across requests --
    the precondition for tiering to onboard (recall) instead of only offload.
    `range_ratio` jitters lengths for a heterogeneous mix; `dataset_path` points
    non-random datasets (sharegpt) at their trace.
    """
    engine_args = engine_args or {}
    if dataset == "prefix_repetition" and 0 < num_prompts < num_prefixes:
        raise ValueError(
            f"prefix_repetition needs num_prompts ({num_prompts}) >= num_prefixes "
            f"({num_prefixes}) so each distinct prefix gets >=1 request"
        )
    concurrencies: list[int | None] = (
        list(max_concurrency) if max_concurrency else [None]
    )
    kvbm_url = (
        f"http://127.0.0.1:{kvbm_metrics_port}/metrics" if kvbm_metrics_port else None
    )
    metrics: list[MetricSample] = []

    ctx_owns_dir = result_dir is None
    tmp_dir = (
        tempfile.mkdtemp(prefix="memos_benchserve_") if ctx_owns_dir else result_dir
    )
    result_path = Path(tmp_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    with VLLMServer(
        model,
        tp=tp,
        port=port,
        engine_args=engine_args,
        env=env,
        log_path=server_log,
    ) as server:
        # Discarded warmup: force graph capture across concurrency buckets before
        # the measured grid so the first real point is warm.
        if warmup_prompts > 0:
            _run_one_bench(
                base_url=server.base_url,
                model=model,
                isl=isl,
                osl=osl,
                num_prompts=warmup_prompts,
                request_rate="inf",
                max_concurrency=None,
                dataset=dataset,
                percentiles=percentiles,
                result_dir=result_path,
                tag="warmup",
                prefix_len=prefix_len,
                range_ratio=range_ratio,
                dataset_path=dataset_path,
                # prefix_repetition requires num_prompts >= num_prefixes (>=1
                # request per distinct prefix); the small warmup pass would else
                # violate it, so clamp the pool to the warmup prompt count.
                num_prefixes=min(num_prefixes, warmup_prompts),
            )

        for rate in request_rates:
            for conc in concurrencies:
                tag = f"isl{isl}_osl{osl}_rate{rate}_conc{conc or 0}"
                # The pollers' contexts scope metric sampling to this bench call:
                # vLLM /metrics for pressure, KVBM :6880 for tier movement.
                poller = PrometheusPoller(server.metrics_url)
                kvbm_poller = KVBMPoller(kvbm_url) if kvbm_url else None
                with ExitStack() as stack:
                    stack.enter_context(poller)
                    if kvbm_poller is not None:
                        stack.enter_context(kvbm_poller)
                    result = _run_one_bench(
                        base_url=server.base_url,
                        model=model,
                        isl=isl,
                        osl=osl,
                        num_prompts=num_prompts,
                        request_rate=rate,
                        max_concurrency=conc,
                        dataset=dataset,
                        percentiles=percentiles,
                        result_dir=result_path,
                        tag=tag,
                        prefix_len=prefix_len,
                        range_ratio=range_ratio,
                        dataset_path=dataset_path,
                        num_prefixes=num_prefixes,
                    )
                parsed = parse_vllm_bench_json(result)
                pressure = poller.summary()
                if kvbm_poller is not None:
                    pressure.update(kvbm_poller.summary())

                ctx: dict[str, Any] = {
                    "request_rate": str(rate),
                    "max_concurrency": conc or 0,
                    "actual_isl": isl,
                    "output_tokens": osl,
                    "num_prompts": num_prompts,
                    "prefix_len": prefix_len,
                    "range_ratio": range_ratio if range_ratio is not None else "0",
                    "num_prefixes": num_prefixes,
                }
                for name, value in {**parsed, **pressure}.items():
                    metrics.append(
                        MetricSample(
                            name=name,
                            value=value,
                            unit=_unit_for(name),
                            context=ctx,
                        )
                    )

    return BenchmarkResult(
        workload="bench_serve",
        hardware=hw.name,
        model=model,
        params={
            "tp": tp,
            "isl": isl,
            "osl": osl,
            "num_prompts": num_prompts,
            "request_rates": list(request_rates),
            "max_concurrency": list(max_concurrency) if max_concurrency else [],
            "dataset": dataset,
            "dataset_path": dataset_path,
            "prefix_len": prefix_len,
            "range_ratio": range_ratio if range_ratio is not None else "0",
            "num_prefixes": num_prefixes,
            "engine_args": engine_args,
        },
        metrics=metrics,
    )
