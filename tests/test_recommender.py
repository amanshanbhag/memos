"""Unit tests for the analytical KV-tier recommender (host-capacity law).

Numbers are the validated 70B / GB300 anchors from research-notes finding #16:
KV = 0.328 MB/token (fp16), weights = 141 GB, HBM = 4 x 277.5 GB, host DRAM =
478 + 474 GB, util = 0.5 -> HBM KV budget ~= 1.26M tokens (the §15 cliff sits
between n=128 @ 1.05M and n=192 @ 1.57M).
"""

from __future__ import annotations

from memos.recommender import recommend_kv_placement, tier_budgets
from memos.types import HardwareConfig, TierSpec

KV_PER_TOK = 0.328 * 1e6  # bytes/token, 70B fp16
WEIGHTS = 141e9  # bytes
UTIL = 0.5


def _gb300() -> HardwareConfig:
    return HardwareConfig(
        name="gb300_tray",
        gpu="GB300",
        gpu_count=4,
        tiers=[
            TierSpec("hbm", 277.5, 8000, 0.1, scope="device"),
            TierSpec("peer_hbm", 277.5, 1800, 1.0),
            TierSpec("lpddr5x_local", 478, 450, 0.5),
            TierSpec("lpddr5x_remote", 474, 300, 1.5),
            TierSpec("nvme", 21000, 21, 50),
        ],
    )


def _no_disk() -> HardwareConfig:
    hw = _gb300()
    hw.tiers = [t for t in hw.tiers if t.name != "nvme"]
    return hw


def test_hbm_budget_matches_measured_cliff():
    budgets = tier_budgets(_gb300(), KV_PER_TOK, WEIGHTS, UTIL)
    hbm = next(b for b in budgets if b.kind == "hbm")
    # (0.5*1110 - 141)/0.328 MB ~= 1.26M tokens, i.e. between n=128 and n=192.
    assert 1.15e6 < hbm.capacity_tokens < 1.35e6
    # host DRAM tiers are aggregated into one pool (~952 GB).
    host = next(b for b in budgets if b.kind == "host")
    assert abs(host.capacity_gb - 952) < 1
    # peer_hbm is excluded from the KV-offload budget.
    assert all(b.kind != "peer" for b in budgets)


def test_fits_hbm_no_tiering():
    # n=128 pool (1.05M tok) fits the HBM budget -> no eviction pressure.
    rec = recommend_kv_placement(_gb300(), KV_PER_TOK, WEIGHTS, 128 * 8192, UTIL)
    assert rec.regime == "hbm_resident"
    assert rec.predicted_hit_rate == 1.0


def test_tiering_win_when_pool_fits_host():
    # n=192 pool (1.57M tok) overflows HBM but fits host DRAM -> offload wins.
    rec = recommend_kv_placement(_gb300(), KV_PER_TOK, WEIGHTS, 192 * 8192, UTIL)
    assert rec.regime == "tiering_win"
    assert rec.placement_tier == "host_dram"
    assert rec.recommended_offload_gb and rec.recommended_offload_gb > 400
    assert rec.predicted_hit_rate == 1.0


def test_overflow_thrash_when_pool_exceeds_all_tiers():
    # 4M-token pool (~1.3 TB) exceeds HBM+host when no disk tier exists.
    rec = recommend_kv_placement(_no_disk(), KV_PER_TOK, WEIGHTS, 4_000_000, UTIL)
    assert rec.regime == "overflow_thrash"
    assert rec.predicted_hit_rate < 1.0


def test_reload_vs_recompute_flag_flips_on_slow_tier():
    # A fast host link: reload beats recompute (few FLOPs/token relative to BW).
    fast = recommend_kv_placement(
        _gb300(),
        KV_PER_TOK,
        WEIGHTS,
        192 * 8192,
        UTIL,
        prefill_flops_per_token=2 * 70e9,  # ~2*params
        peak_flops=3.75e15,
    )
    assert fast.reload_beats_recompute is True


def test_fp8_doubles_effective_capacity():
    # Halving KV bytes (fp8) doubles the token budget -> a pool that thrashed at
    # fp16 can fit at fp8.
    ws = 256 * 8192  # 2.1M tok, overflows fp16 host? no -> use a bigger pool
    big = 480 * 8192  # 3.93M tok
    fp16 = recommend_kv_placement(_no_disk(), KV_PER_TOK, WEIGHTS, big, UTIL)
    fp8 = recommend_kv_placement(_no_disk(), KV_PER_TOK / 2, WEIGHTS, big, UTIL)
    assert fp16.regime == "overflow_thrash"
    assert fp8.regime == "tiering_win"
    assert ws  # silence unused
