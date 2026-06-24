from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from memos.types import MetricSample


class MetricCollector(ABC):
    """Observes generation events and produces metric summaries."""

    @abstractmethod
    def on_generate_start(self, request_id: str, context_length: int) -> None:
        """Called before each generation request."""
        ...

    @abstractmethod
    def on_generate_end(
        self,
        request_id: str,
        tokens_generated: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        """Called after each generation request completes."""
        ...

    @abstractmethod
    def summarize(self) -> list[MetricSample]:
        """Return all collected metric samples."""
        ...

    def reset(self) -> None:
        """Clear collected data for a fresh run."""
        pass
