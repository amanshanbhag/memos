from __future__ import annotations

from typing import Any

from memos.metrics.base import MetricCollector
from memos.types import MetricSample


class BytesPerTokenCollector(MetricCollector):
    """Tracks total bytes of data movement per generated token (BPT).

    Requires the runner to report bytes_moved in on_generate_end kwargs.
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
        bytes_moved = kwargs.get("bytes_moved", 0)
        if tokens_generated > 0:
            bpt = bytes_moved / tokens_generated
            self._samples.append(
                MetricSample(
                    name="bytes_per_token",
                    value=bpt,
                    unit="bytes",
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
