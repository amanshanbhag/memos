from __future__ import annotations

import time
from typing import Any

from memos.runners.base import GenerateParams, GenerateResult, Runner
from memos.types import HardwareConfig


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
        self, prompts: list[str], params: GenerateParams
    ) -> list[GenerateResult]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            **params.extra,
        )

        start = time.perf_counter()
        outputs = self._llm.generate(prompts, sampling_params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Pull cache metrics snapshot after generation
        metrics = self.get_metrics()

        results = []
        for output in outputs:
            results.append(
                GenerateResult(
                    prompt_tokens=len(output.prompt_token_ids),
                    generated_tokens=len(output.outputs[0].token_ids),
                    duration_ms=elapsed_ms / len(outputs),
                    text=output.outputs[0].text,
                    extra={
                        "kv_cache_usage": metrics.get("vllm:kv_cache_usage_perc", 0.0),
                        "prefix_cache_hits": metrics.get("vllm:prefix_cache_hits", 0.0),
                        "prefix_cache_queries": metrics.get(
                            "vllm:prefix_cache_queries", 0.0
                        ),
                    },
                )
            )
        return results

    def shutdown(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
