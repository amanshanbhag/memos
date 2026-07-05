from __future__ import annotations

import threading
import time
from typing import Any

from memos.runners.base import GenerateParams, GenerateResult, Prompt, Runner
from memos.types import HardwareConfig


class _MetricSampler:
    """Polls engine metrics on a background thread during a blocking generate().

    The vLLM V1 engine runs in separate processes; `generate()` blocks the
    calling thread, so end-of-call counters never reveal the *in-flight* state
    (e.g. `kv_cache_usage_perc` reads ~0 once the cache is freed). This sampler
    captures the running batch and per-tier cache pressure while generation is
    actually happening, so the roofline can be positioned at the MEASURED
    average running batch rather than the nominal submitted count.

    Fully defensive: any error in polling is swallowed so it can never break a
    benchmark run.
    """

    def __init__(self, runner: "VLLMRunner", interval_s: float = 0.1) -> None:
        self._runner = runner
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running: list[float] = []
        self._kv: list[float] = []
        self._gpu: list[float] = []
        self._cpu: list[float] = []

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                m = self._runner.get_metrics()
                r = m.get("vllm:num_requests_running")
                if r is not None:
                    self._running.append(float(r))
                for key, buf in (
                    ("vllm:kv_cache_usage_perc", self._kv),
                    ("vllm:gpu_cache_usage_perc", self._gpu),
                    ("vllm:cpu_cache_usage_perc", self._cpu),
                ):
                    v = m.get(key)
                    if v is not None:
                        buf.append(float(v))
            except Exception:
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> "_MetricSampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def summary(self) -> dict[str, float]:
        active = [x for x in self._running if x > 0]
        return {
            "running_batch_avg": (sum(active) / len(active)) if active else 0.0,
            "running_batch_peak": max(self._running) if self._running else 0.0,
            "kv_cache_usage_peak": max(self._kv) if self._kv else 0.0,
            "gpu_cache_usage_peak": max(self._gpu) if self._gpu else 0.0,
            "cpu_cache_usage_peak": max(self._cpu) if self._cpu else 0.0,
        }


class VLLMRunner(Runner):
    """vLLM-based inference runner."""

    def __init__(self) -> None:
        self._llm = None
        self._model_name = ""

    def setup(self, model: str, hw: HardwareConfig, **kwargs: Any) -> None:
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install -e '.[vllm]'")

        self._model_name = model

        tp = kwargs.pop("tensor_parallel_size", hw.gpu_count)
        gpu_util = kwargs.pop("gpu_memory_utilization", 0.9)
        prefix_cache = kwargs.pop("enable_prefix_caching", False)

        kwargs.setdefault("kv_cache_metrics", True)
        kwargs.setdefault("enable_mfu_metrics", True)

        self._llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            enable_prefix_caching=prefix_cache,
            disable_log_stats=False,
            generation_config="vllm",
            **kwargs,
        )

    @property
    def tokenizer(self):
        """Return the underlying tokenizer for exact prompt construction."""
        if self._llm is None:
            raise RuntimeError("Runner not set up; call setup() first")
        return self._llm.get_tokenizer()

    def get_metrics(self) -> dict[str, float]:
        if self._llm is None:
            return {}
        raw = self._llm.get_metrics()
        result = {}
        for metric in raw:
            if not hasattr(metric, "value"):
                continue
            name = metric.name
            if hasattr(metric, "labels") and metric.labels:
                extra = {
                    k: v
                    for k, v in metric.labels.items()
                    if k not in ("model_name", "engine")
                }
                if extra:
                    suffix = "_".join(f"{v}" for v in extra.values())
                    name = f"{name}_{suffix}"
            result[name] = metric.value
        return result

    def generate(
        self, prompts: list[Prompt], params: GenerateParams
    ) -> list[GenerateResult]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=params.max_tokens,
            min_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            ignore_eos=True,
            **params.extra,
        )

        text_prompts: list[str] = []
        token_id_prompts: list[dict] = []
        uses_token_ids = False

        for p in prompts:
            if isinstance(p, list):
                token_id_prompts.append({"prompt_token_ids": p})
                uses_token_ids = True
            else:
                text_prompts.append(p)

        with _MetricSampler(self) as sampler:
            start = time.perf_counter()
            if uses_token_ids:
                outputs = self._llm.generate(token_id_prompts, sampling_params)
            else:
                outputs = self._llm.generate(text_prompts, sampling_params)
            elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = self.get_metrics()
        sampled = sampler.summary()

        results = []
        for output in outputs:
            results.append(
                GenerateResult(
                    prompt_tokens=len(output.prompt_token_ids),
                    generated_tokens=len(output.outputs[0].token_ids),
                    duration_ms=elapsed_ms / len(outputs),
                    text=output.outputs[0].text,
                    extra={
                        # In-flight cache usage sampled DURING generation (the
                        # end-of-call gauge reads ~0 after the cache is freed).
                        "kv_cache_usage": sampled["kv_cache_usage_peak"]
                        or metrics.get("vllm:kv_cache_usage_perc", 0.0),
                        "prefix_cache_hits": metrics.get("vllm:prefix_cache_hits", 0.0),
                        "prefix_cache_queries": metrics.get(
                            "vllm:prefix_cache_queries", 0.0
                        ),
                        # Pressure/spill signals: num_preemptions rises when KV
                        # overflows HBM (vLLM preempts+recomputes or swaps); cpu
                        # cache usage rises when KV spills to host DRAM.
                        "num_preemptions": metrics.get(
                            "vllm:num_preemptions_total",
                            metrics.get("vllm:num_preemptions", 0.0),
                        ),
                        "gpu_cache_usage": sampled["gpu_cache_usage_peak"]
                        or metrics.get(
                            "vllm:gpu_cache_usage_perc",
                            metrics.get("vllm:kv_cache_usage_perc", 0.0),
                        ),
                        "cpu_cache_usage": sampled["cpu_cache_usage_peak"]
                        or metrics.get("vllm:cpu_cache_usage_perc", 0.0),
                        # Measured running batch (for correct roofline positioning).
                        "running_batch_avg": sampled["running_batch_avg"],
                        "running_batch_peak": sampled["running_batch_peak"],
                    },
                )
            )
        return results

    def shutdown(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
