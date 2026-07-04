from __future__ import annotations

from typing import Any

from memos.metrics.base import MetricCollector
from memos.types import MetricSample


class ThroughputCollector(MetricCollector):
    """Tracks tokens/sec and kv cache usage from each generation."""

    def __init__(self) -> None:
        self._samples: list[MetricSample] = []
        self._current_context = 0

    def on_generate_start(self, request_id: str, context_length: int) -> None:
        self._current_context = context_length

    def on_generate_end(
        self,
        request_id: str,
        tokens_generated: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        ctx: dict[str, Any] = {
            "context_length": self._current_context,
            "request_id": request_id,
        }
        if "output_tokens" in kwargs:
            ctx["output_tokens"] = kwargs["output_tokens"]
        if "actual_isl" in kwargs:
            ctx["actual_isl"] = kwargs["actual_isl"]
        if "batch" in kwargs:
            ctx["batch"] = kwargs["batch"]

        if duration_ms > 0 and tokens_generated > 0:
            tps = tokens_generated / (duration_ms / 1000)
            self._samples.append(
                MetricSample(
                    name="tokens_per_sec", value=tps, unit="tokens/s", context=ctx
                )
            )

        self._samples.append(
            MetricSample(name="duration_ms", value=duration_ms, unit="ms", context=ctx)
        )

        kv_usage = kwargs.get("kv_cache_usage", None)
        if kv_usage is not None:
            self._samples.append(
                MetricSample(
                    name="kv_cache_usage", value=kv_usage, unit="ratio", context=ctx
                )
            )

        # Spill/pressure signals for the multi-tier roofline experiments.
        for name, unit in (
            ("num_preemptions", "count"),
            ("cpu_cache_usage", "ratio"),
        ):
            val = kwargs.get(name, None)
            if val is not None:
                self._samples.append(
                    MetricSample(name=name, value=val, unit=unit, context=ctx)
                )

    def summarize(self) -> list[MetricSample]:
        return self._samples

    def reset(self) -> None:
        self._samples = []
