"""Synthetic reuse-workload generation for the serving benchmark (Track A3).

`vllm bench`'s `prefix_repetition` dataset repeats each of N distinct prefixes
UNIFORMLY (num_prompts // N times each). Real agentic/serving traffic is SKEWED:
a few "hot" contexts are reused heavily while a long tail is touched rarely. Skew
is exactly the regime where a SMALL fast tier captures MOST of the reuse -- i.e.
where graduated, popularity-aware placement beats the all-or-nothing capacity law
(research notes: "North star: use ALL tiers without losing perf").

vLLM has no skew knob, so we materialize a Zipfian-popularity workload into a
`custom` dataset (JSONL of `{"prompt": ...}` lines) and point `vllm bench serve
--dataset-name custom` at it. Because tokenization is a deterministic function of
the text, writing the SAME prefix text on many lines yields identical prefix
tokens on every request -> vLLM prefix caching reuses the shared blocks (the
precondition for KV tiering to recall instead of recompute). A per-request unique
suffix (after a newline separator that forces a token boundary) keeps requests
distinct so decode still happens and only the PREFIX is shared.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class _Tokenizer(Protocol):
    """The minimal tokenizer surface we need (HF AutoTokenizer satisfies it)."""

    vocab_size: int
    all_special_ids: list[int]

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass
class ZipfDatasetStats:
    """Provenance for a generated Zipfian reuse dataset (recorded in results)."""

    num_prefixes: int
    prefix_len_target: int
    prefix_len_min: int
    prefix_len_max: int
    prefix_len_mean: float
    suffix_len_target: int
    num_prompts: int
    zipf_s: float
    distinct_prefixes_used: int
    top1_share: float  # fraction of requests hitting the single hottest prefix
    top10pct_share: float  # fraction hitting the hottest 10% of prefixes
    path: str


def zipf_weights(n: int, s: float) -> list[float]:
    """Normalized Zipf(s) weights over ranks 1..n. s=0 -> uniform."""
    if n <= 0:
        return []
    raw = [1.0 / (rank**s) for rank in range(1, n + 1)]
    total = sum(raw)
    return [w / total for w in raw]


def _vocab_pool(tokenizer: _Tokenizer) -> list[int]:
    """Usable (non-special) token ids to synthesize filler text from."""
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    return [i for i in range(int(tokenizer.vocab_size)) if i not in special]


def _exact_len_text(
    tokenizer: _Tokenizer, target: int, rng: random.Random, pool: list[int]
) -> tuple[str, int]:
    """Build text that re-tokenizes to ~`target` tokens (stable across reuse).

    Random token ids are decoded to text; because decode/encode need not round-
    trip, we iterate a few times (trim if long, extend if short) to converge on
    the target length. Returns (text, actual_token_len).
    """
    ids = [rng.choice(pool) for _ in range(target)]
    text = tokenizer.decode(ids)
    for _ in range(6):
        toks = tokenizer.encode(text, add_special_tokens=False)
        n = len(toks)
        if n == target:
            return text, target
        if n > target:
            text = tokenizer.decode(toks[:target])
        else:
            extra = tokenizer.decode([rng.choice(pool) for _ in range(target - n)])
            text = f"{text} {extra}"
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def generate_zipf_prefix_dataset(
    path: str | Path,
    tokenizer: _Tokenizer,
    num_prefixes: int,
    prefix_len: int,
    suffix_len: int,
    num_prompts: int,
    zipf_s: float = 1.0,
    seed: int = 0,
) -> ZipfDatasetStats:
    """Write a Zipfian-popularity `custom` JSONL and return its provenance.

    A pool of `num_prefixes` DISTINCT prefixes (each ~`prefix_len` tokens) is
    reused across `num_prompts` requests with popularity ~ 1/rank^`zipf_s`
    (`zipf_s=0` reproduces uniform, matching prefix_repetition). Each request is
    `hot_prefix + "\\n" + unique_suffix` (~`suffix_len` suffix tokens); the
    newline forces a token boundary so the prefix's KV blocks match exactly.
    """
    if num_prefixes <= 0 or prefix_len <= 0 or num_prompts <= 0:
        raise ValueError("num_prefixes, prefix_len, num_prompts must all be > 0")

    rng = random.Random(seed)
    pool = _vocab_pool(tokenizer)
    if not pool:
        raise ValueError("tokenizer has no usable (non-special) tokens")

    prefixes: list[str] = []
    actual_lens: list[int] = []
    for _ in range(num_prefixes):
        text, n = _exact_len_text(tokenizer, prefix_len, rng, pool)
        prefixes.append(text)
        actual_lens.append(n)

    weights = zipf_weights(num_prefixes, zipf_s)
    # sample the hot-prefix index for every request up front (for stats + write)
    picks = rng.choices(range(num_prefixes), weights=weights, k=num_prompts)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for req_id, idx in enumerate(picks):
            # unique suffix keeps each request distinct (forces decode) while the
            # prefix is shared; include req_id so no two suffixes collide.
            suffix_ids = [rng.choice(pool) for _ in range(max(suffix_len, 1))]
            suffix = tokenizer.decode(suffix_ids)
            prompt = f"{prefixes[idx]}\n{req_id} {suffix}"
            f.write(json.dumps({"prompt": prompt}) + "\n")

    counts = [0] * num_prefixes
    for idx in picks:
        counts[idx] += 1
    counts_sorted = sorted(counts, reverse=True)
    top1 = counts_sorted[0] / num_prompts if num_prompts else 0.0
    top10pct_n = max(1, num_prefixes // 10)
    top10pct = sum(counts_sorted[:top10pct_n]) / num_prompts if num_prompts else 0.0

    stats = ZipfDatasetStats(
        num_prefixes=num_prefixes,
        prefix_len_target=prefix_len,
        prefix_len_min=min(actual_lens),
        prefix_len_max=max(actual_lens),
        prefix_len_mean=sum(actual_lens) / len(actual_lens),
        suffix_len_target=suffix_len,
        num_prompts=num_prompts,
        zipf_s=zipf_s,
        distinct_prefixes_used=sum(1 for c in counts if c > 0),
        top1_share=top1,
        top10pct_share=top10pct,
        path=str(out_path),
    )
    return stats


def stats_to_dict(stats: ZipfDatasetStats) -> dict[str, Any]:
    return asdict(stats)
