from __future__ import annotations

from typing import Any

from memos.metrics.base import MetricCollector
from memos.types import MetricSample


class CAFCollector(MetricCollector):
    """Tracks Capacity Amplification Factor - effective memory / physical HBM.

    Requires the runner to report total_kv_bytes and hbm_capacity_bytes
    in on_generate_end kwargs.
    """

    def __init__(self) -> None:
        self._samples: list[MetricSample] = []

    def on_generate_start(self, request_id: str, context_length: int) -> None:
        pass

    def on_generate_end(
        self,
        request_id: str,
        tokens_generated: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        total_kv = kwargs.get("total_kv_bytes", 0)
        hbm_capacity = kwargs.get("hbm_capacity_bytes", 0)
        if hbm_capacity > 0:
            caf = total_kv / hbm_capacity
            self._samples.append(
                MetricSample(
                    name="capacity_amplification_factor",
                    value=caf,
                    unit="ratio",
                    context={"request_id": request_id},
                )
            )

    def summarize(self) -> list[MetricSample]:
        return self._samples

    def reset(self) -> None:
        self._samples = []
