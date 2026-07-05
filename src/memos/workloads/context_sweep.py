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
        batch: int = 1,
        batch_mode: str = "static",
        queue_depth: int = 4,
    ) -> None:
        self._isls = isls or DEFAULT_ISLS
        self._osls = osls or DEFAULT_OSLS
        self._repeats = repeats
        self._cache_mode = cache_mode
        self._batch = max(int(batch), 1)
        if batch_mode not in ("static", "concurrency", "saturated"):
            raise ValueError(
                "batch_mode must be 'static', 'concurrency', or 'saturated', "
                f"got {batch_mode!r}"
            )
        self._batch_mode = batch_mode
        self._queue_depth = max(int(queue_depth), 1)

    def name(self) -> str:
        return "context_sweep"

    def description(self) -> str:
        return (
            f"ISL x OSL sweep: ISL={self._isls}, OSL={self._osls}, "
            f"batch={self._batch} ({self._batch_mode}), "
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
        batch = self._batch
        warmup_iters = 2 if batch > 1 else 5

        for isl, osl in combos:
            params = GenerateParams(max_tokens=osl)
            for _ in range(warmup_iters):
                prompts = self._make_batch(isl, tokenizer, unique=True)
                runner.generate(prompts, params)

        for isl, osl in combos:
            params = GenerateParams(max_tokens=osl)
            unique = self._cache_mode == "cold"

            if self._batch_mode == "saturated":
                # Deep queue (batch * queue_depth prompts) submitted at once so
                # the scheduler keeps `batch` sequences running (pin running
                # batch via max_num_seqs=batch at engine setup). One warmup wave
                # dropped; the sampler records the ACTUAL running batch for the
                # roofline. Throughput is the sustained aggregate over repeats.
                n = max(batch * self._queue_depth, batch)
                runner.generate(self._make_batch(isl, tokenizer, unique, batch), params)
                total_tokens = 0.0
                total_wall_ms = 0.0
                actual_isl = isl
                extra = {}
                for _ in range(self._repeats):
                    prompts = self._make_batch(isl, tokenizer, unique, n)
                    results = runner.generate(prompts, params)
                    total_tokens += sum(r.generated_tokens for r in results)
                    total_wall_ms += _wall_ms(results)
                    actual_isl = results[0].prompt_tokens
                    extra = results[0].extra
                request_id = f"isl{isl}_osl{osl}_b{batch}_saturated"
                for c in collectors:
                    c.on_generate_start(request_id, actual_isl)
                    c.on_generate_end(
                        request_id,
                        total_tokens,
                        total_wall_ms,
                        output_tokens=osl,
                        actual_isl=actual_isl,
                        batch=batch,
                        **extra,
                    )
                continue

            if self._batch_mode == "concurrency":
                # Sustained load: one warmup wave (dropped) then measured waves,
                # aggregated into a single steady-state throughput sample.
                runner.generate(self._make_batch(isl, tokenizer, unique), params)
                total_tokens = 0.0
                total_wall_ms = 0.0
                actual_isl = isl
                extra: dict = {}
                for wave in range(self._repeats):
                    prompts = self._make_batch(isl, tokenizer, unique)
                    results = runner.generate(prompts, params)
                    total_tokens += sum(r.generated_tokens for r in results)
                    total_wall_ms += _wall_ms(results)
                    actual_isl = results[0].prompt_tokens
                    extra = results[0].extra
                request_id = f"isl{isl}_osl{osl}_b{batch}_sustained"
                for c in collectors:
                    c.on_generate_start(request_id, actual_isl)
                    c.on_generate_end(
                        request_id,
                        total_tokens,
                        total_wall_ms,
                        output_tokens=osl,
                        actual_isl=actual_isl,
                        batch=batch,
                        **extra,
                    )
                continue

            # Static batch: exactly `batch` requests per generate call, per repeat.
            for repeat in range(self._repeats):
                request_id = f"isl{isl}_osl{osl}_b{batch}_r{repeat}"
                prompts = self._make_batch(isl, tokenizer, unique)
                actual_isl = len(prompts[0]) if isinstance(prompts[0], list) else isl

                for c in collectors:
                    c.on_generate_start(request_id, actual_isl)

                results = runner.generate(prompts, params)
                total_tokens = sum(r.generated_tokens for r in results)
                wall_ms = _wall_ms(results)

                for c in collectors:
                    c.on_generate_end(
                        request_id,
                        total_tokens,
                        wall_ms,
                        output_tokens=osl,
                        actual_isl=results[0].prompt_tokens,
                        batch=batch,
                        **results[0].extra,
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
                "batch": self._batch,
                "batch_mode": self._batch_mode,
            },
            metrics=all_metrics,
        )

    def _make_batch(
        self, isl: int, tokenizer, unique: bool, count: int | None = None
    ) -> list[Prompt]:
        """Build `count` (default `batch`) prompts of exactly `isl` tokens.

        Each prompt is independently randomized so cold-cache runs don't share
        prefixes across the batch (which would let prefix caching interfere).
        """
        n = self._batch if count is None else max(int(count), 1)
        return [_make_exact_prompt(isl, tokenizer, unique=unique) for _ in range(n)]


def _wall_ms(results) -> float:
    """Reconstruct wall-clock time for a batched generate call.

    VLLMRunner reports per-result duration_ms = wall_ms / batch_size, so summing
    across the batch recovers the true wall-clock elapsed time. Aggregate
    throughput is then total_generated_tokens / wall_seconds.
    """
    return sum(r.duration_ms for r in results)


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
