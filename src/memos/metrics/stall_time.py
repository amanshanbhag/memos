from __future__ import annotations

from typing import Any

from memos.metrics.base import MetricCollector
from memos.types import MetricSample


class StallTimeCollector(MetricCollector):
    """Tracks microseconds spent waiting for tier-crossing transfers per token (STPT).

    Requires the runner to report stall_us in on_generate_end kwargs.
    """

    def __init__(self) -> None:
        self._samples: list[MetricSample] = []

    def on_generate_start(self, request_id: str, context_length: int) -> None:
        self._current_context = context_length

    def on_generate_end(
        self,
        request_id: str,
        tokens_generated: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        stall_us = kwargs.get("stall_us", 0)
        if tokens_generated > 0:
            stpt = stall_us / tokens_generated
            self._samples.append(
                MetricSample(
                    name="stall_time_per_token",
                    value=stpt,
                    unit="us",
                    context={
                        "context_length": self._current_context,
                        "request_id": request_id,
                    },
                )
            )

    def summarize(self) -> list[MetricSample]:
        return self._samples

    def reset(self) -> None:
        self._samples = []
