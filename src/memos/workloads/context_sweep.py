from __future__ import annotations

from memos.metrics.base import MetricCollector
from memos.runners.base import GenerateParams, Runner
from memos.types import BenchmarkResult, HardwareConfig, MetricSample

DEFAULT_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]


class ContextSweep(Workload):
    """Sweep context lengths, measuring how memory behavior changes with scale.

    Generates a fixed number of output tokens at each context length.
    Reveals the relationship between context size and bytes/token,
    stall time, and throughput.
    """

    def __init__(
        self,
        context_lengths: list[int] | None = None,
        output_tokens: int = 128,
        repeats: int = 3,
        cache_mode: str = "cold",  # "cold" = unique prompts, "warm" = reuse prompts
    ) -> None:
        self._context_lengths = context_lengths or DEFAULT_CONTEXT_LENGTHS
        self._output_tokens = output_tokens
        self._repeats = repeats

    def name(self) -> str:
        return "context_sweep"

    def description(self) -> str:
        return f"Context length sweep: {self._context_lengths}, {self._repeats} repeats each"

    def run(
        self,
        runner: Runner,
        hw: HardwareConfig,
        collectors: list[MetricCollector],
    ) -> BenchmarkResult:
        all_metrics: list[MetricSample] = []
        params = GenerateParams(max_tokens=self._output_tokens)

        for ctx_len in self._context_lengths:
            prompt = _make_prompt(ctx_len)

            for repeat in range(self._repeats):
                request_id = f"ctx{ctx_len}_r{repeat}"

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
            model="",  # filled in by CLI
            params={
                "context_lengths": self._context_lengths,
                "output_tokens": self._output_tokens,
                "repeats": self._repeats,
            },
            metrics=all_metrics,
        )


def _make_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length.

    Uses a simple repeating pattern. A real implementation would use
    a tokenizer for precise token counting.
    """
    word = "the quick brown fox jumps over the lazy dog "
    approx_chars = target_tokens * 4  # rough chars-per-token estimate
    return (word * (approx_chars // len(word) + 1))[:approx_chars]
