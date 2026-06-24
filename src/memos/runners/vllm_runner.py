from __future__ import annotations

from typing import Any

from memos.runners.base import GenerateParams, GenerateResult, Runner
from memos.types import HardwareConfig


class VLLMRunner(Runner):
    """vLLM-based inference runner."""

    def setup(self, model: str, hw: HardwareConfig, **kwargs: Any) -> None:
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install -e '.[vllm]'")
        raise NotImplementedError("TODO: configure and instantiate vllm.LLM")

    def generate(
        self, prompts: list[str], params: GenerateParams
    ) -> list[GenerateResult]:
        raise NotImplementedError("TODO: call self.llm.generate()")

    def get_metrics(self) -> dict[str, float]:
        raise NotImplementedError("TODO: extract cache stats from vLLM internals")

    def shutdown(self) -> None:
        raise NotImplementedError("TODO: cleanup")
