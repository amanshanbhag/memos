"""Unit tests for the Zipfian skewed-reuse dataset generator (Track A3).

Uses a fake word-tokenizer that round-trips exactly (decode(encode(x)) == x for
space-joined `w<id>` tokens), so prefix lengths land exactly on target and reuse
is easy to assert.
"""

from __future__ import annotations

import json

from memos.serving.datasets import generate_zipf_prefix_dataset, zipf_weights


class FakeTokenizer:
    vocab_size = 200
    all_special_ids = [0, 1]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(t[1:]) for t in text.split() if t.startswith("w")]

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w{i}" for i in ids)


def test_zipf_weights_uniform_and_skewed():
    uni = zipf_weights(5, 0.0)
    assert abs(sum(uni) - 1.0) < 1e-9
    assert max(uni) - min(uni) < 1e-9  # all equal
    skew = zipf_weights(5, 1.5)
    assert abs(sum(skew) - 1.0) < 1e-9
    assert skew[0] > skew[-1] * 5  # rank-1 dominates


def test_generated_file_structure(tmp_path):
    p = tmp_path / "zipf.jsonl"
    stats = generate_zipf_prefix_dataset(
        p,
        FakeTokenizer(),
        num_prefixes=8,
        prefix_len=32,
        suffix_len=4,
        num_prompts=200,
        zipf_s=1.0,
        seed=0,
    )
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 200
    for ln in lines:
        obj = json.loads(ln)
        assert "prompt" in obj and obj["prompt"]
    assert stats.distinct_prefixes_used <= 8
    assert stats.num_prompts == 200


def test_exact_prefix_length_with_roundtrip_tokenizer(tmp_path):
    stats = generate_zipf_prefix_dataset(
        tmp_path / "z.jsonl",
        FakeTokenizer(),
        num_prefixes=4,
        prefix_len=64,
        suffix_len=8,
        num_prompts=50,
        zipf_s=1.0,
    )
    assert stats.prefix_len_min == 64
    assert stats.prefix_len_max == 64


def test_skew_concentrates_more_than_uniform(tmp_path):
    uni = generate_zipf_prefix_dataset(
        tmp_path / "u.jsonl",
        FakeTokenizer(),
        num_prefixes=16,
        prefix_len=16,
        suffix_len=4,
        num_prompts=2000,
        zipf_s=0.0,
        seed=1,
    )
    skew = generate_zipf_prefix_dataset(
        tmp_path / "s.jsonl",
        FakeTokenizer(),
        num_prefixes=16,
        prefix_len=16,
        suffix_len=4,
        num_prompts=2000,
        zipf_s=1.4,
        seed=1,
    )
    # skew should concentrate mass on the hot prefixes.
    assert skew.top1_share > uni.top1_share * 2
    assert skew.top10pct_share > uni.top10pct_share


def test_prefix_actually_reused(tmp_path):
    p = tmp_path / "z.jsonl"
    generate_zipf_prefix_dataset(
        p,
        FakeTokenizer(),
        num_prefixes=8,
        prefix_len=16,
        suffix_len=4,
        num_prompts=200,
        zipf_s=1.0,
    )
    prefixes = {
        json.loads(ln)["prompt"].split("\n", 1)[0] for ln in p.read_text().splitlines()
    }
    # 200 requests share a pool of <=8 distinct prefixes -> heavy reuse.
    assert len(prefixes) <= 8


def test_rejects_bad_args(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        generate_zipf_prefix_dataset(
            tmp_path / "x.jsonl",
            FakeTokenizer(),
            num_prefixes=0,
            prefix_len=16,
            suffix_len=4,
            num_prompts=10,
        )
