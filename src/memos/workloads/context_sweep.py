from __future__ import annotations

import random
import string

from memos.metrics.base import MetricCollector
from memos.runners.base import GenerateParams, Prompt, Runner
from memos.types import BenchmarkResult, HardwareConfig, MetricSample
from memos.workloads.base import Workload

DEFAULT_ISLS = [4096, 8192, 16384, 32768, 65536, 128000]
DEFAULT_OSLS = [128]


class ContextSweep(Workload):

    def __init__(
        self,
        isls: list[int] | None = None,
        osls: list[int] | None = None,
        repeats: int = 3,
        cache_mode: str = "cold",
    ) -> None:
        self._isls = isls or DEFAULT_ISLS
        self._osls = osls or DEFAULT_OSLS
        self._repeats = repeats
        self._cache_mode = cache_mode

    def name(self) -> str:
        return "context_sweep"

    def description(self) -> str:
        return (
            f"ISL x OSL sweep: ISL={self._isls}, OSL={self._osls}, "
            f"{self._repeats} repeats each"
        )

    def run(
        self,
        runner: Runner,
        hw: HardwareConfig,
        collectors: list[MetricCollector],
        model: str = "",
    ) -> BenchmarkResult:
        all_metrics: list[MetricSample] = []
        combos = [(isl, osl) for isl in self._isls for osl in self._osls]

        tokenizer = runner.tokenizer

        for isl, osl in combos:
            params = GenerateParams(max_tokens=osl)
            for _ in range(5):
                prompt = _make_exact_prompt(isl, tokenizer, unique=True)
                runner.generate([prompt], params)

        for isl, osl in combos:
            params = GenerateParams(max_tokens=osl)
            for repeat in range(self._repeats):
                request_id = f"isl{isl}_osl{osl}_r{repeat}"
                unique = self._cache_mode == "cold"
                prompt = _make_exact_prompt(isl, tokenizer, unique=unique)

                actual_isl = len(prompt) if isinstance(prompt, list) else isl

                for c in collectors:
                    c.on_generate_start(request_id, actual_isl)

                results = runner.generate([prompt], params)
                result = results[0]

                for c in collectors:
                    c.on_generate_end(
                        request_id,
                        result.generated_tokens,
                        result.duration_ms,
                        output_tokens=osl,
                        actual_isl=result.prompt_tokens,
                        **result.extra,
                    )

        for c in collectors:
            all_metrics.extend(c.summarize())

        return BenchmarkResult(
            workload=self.name(),
            hardware=hw.name,
            model=model,
            params={
                "isls": self._isls,
                "osls": self._osls,
                "repeats": self._repeats,
                "cache_mode": self._cache_mode,
            },
            metrics=all_metrics,
        )


def _make_exact_prompt(target_tokens: int, tokenizer, unique: bool = True) -> list[int]:
    """Generate a prompt with exactly target_tokens token IDs.

    Overshoots with raw text, tokenizes, then truncates to the exact length.
    Returns token IDs directly to bypass any tokenizer ambiguity.
    """
    overshoot = int(target_tokens * 5)
    if unique:
        chars = string.ascii_lowercase + string.digits + " "
        raw = "".join(random.choices(chars, k=overshoot))
    else:
        words = "the quick brown fox jumps over the lazy dog "
        raw = (words * (overshoot // len(words) + 1))[:overshoot]

    token_ids = tokenizer.encode(raw)

    if len(token_ids) < target_tokens:
        extra_raw = "".join(random.choices(string.ascii_lowercase, k=overshoot))
        token_ids.extend(tokenizer.encode(extra_raw))

    return token_ids[:target_tokens]
