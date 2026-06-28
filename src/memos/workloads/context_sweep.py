from __future__ import annotations

import random
import string

from memos.metrics.base import MetricCollector
from memos.runners.base import GenerateParams, Runner
from memos.types import BenchmarkResult, HardwareConfig, MetricSample
from memos.workloads.base import Workload

DEFAULT_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]


class ContextSweep(Workload):

    def __init__(
        self,
        context_lengths: list[int] | None = None,
        output_tokens: int = 128,
        repeats: int = 3,
        cache_mode: str = "cold",
    ) -> None:
        self._context_lengths = context_lengths or DEFAULT_CONTEXT_LENGTHS
        self._output_tokens = output_tokens
        self._repeats = repeats
        self._cache_mode = cache_mode

    def name(self) -> str:
        return "context_sweep"

    def description(self) -> str:
        return f"Context length sweep: {self._context_lengths}, {self._repeats} repeats each"

    def run(
        self,
        runner: Runner,
        hw: HardwareConfig,
        collectors: list[MetricCollector],
        model: str = "",
    ) -> BenchmarkResult:
        all_metrics: list[MetricSample] = []
        params = GenerateParams(max_tokens=self._output_tokens)

        # Warmup: 5 throwaway requests per context length to trigger compilation
        for ctx_len in self._context_lengths:
            for _ in range(5):
                prompt = _make_prompt_unique(ctx_len)
                runner.generate([prompt], params)

        for ctx_len in self._context_lengths:
            for repeat in range(self._repeats):
                request_id = f"ctx{ctx_len}_r{repeat}"

                if self._cache_mode == "cold":
                    prompt = _make_prompt_unique(ctx_len)
                else:
                    prompt = _make_prompt(ctx_len)

                for c in collectors:
                    c.on_generate_start(request_id, ctx_len)

                results = runner.generate([prompt], params)
                result = results[0]

                for c in collectors:
                    c.on_generate_end(
                        request_id,
                        result.generated_tokens,
                        result.duration_ms,
                        **result.extra,
                    )

        for c in collectors:
            all_metrics.extend(c.summarize())

        return BenchmarkResult(
            workload=self.name(),
            hardware=hw.name,
            model=model,
            params={
                "context_lengths": self._context_lengths,
                "output_tokens": self._output_tokens,
                "repeats": self._repeats,
                "cache_mode": self._cache_mode,
            },
            metrics=all_metrics,
        )


def _make_prompt(target_tokens: int) -> str:
    words = "the quick brown fox jumps over the lazy dog "
    approx_chars = target_tokens * 4
    return (words * (approx_chars // len(words) + 1))[:approx_chars]


def _make_prompt_unique(target_tokens: int) -> str:
    """Generate a fully random prompt so content is unpredictable."""
    chars = string.ascii_lowercase + string.digits + " "
    return "".join(random.choices(chars, k=target_tokens * 4))
