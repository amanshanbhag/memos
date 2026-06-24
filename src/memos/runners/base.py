from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from memos.types import HardwareConfig


@dataclass
class GenerateParams:
    """Parameters for a generation request."""

    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResult:
    """Result from a single generation request."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    duration_ms: float = 0.0
    text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Runner(ABC):
    """Abstract interface for an inference engine."""

    @abstractmethod
    def setup(self, model: str, hw: HardwareConfig, **kwargs: Any) -> None:
        """Load the model and prepare for generation."""
        ...

    @abstractmethod
    def generate(
        self, prompts: list[str], params: GenerateParams
    ) -> list[GenerateResult]:
        """Run generation on a batch of prompts."""
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Return engine-level metrics (cache utilization, evictions, etc.)."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources."""
        ...
