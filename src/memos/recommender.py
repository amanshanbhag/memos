"""Analytical KV-tier placement recommender (Track D1).

Operationalizes the HOST-CAPACITY LAW discovered empirically (research notes
finding #16, sweep 07_12 §15): for a workload with a REUSED KV working set,

  - if the working set fits in HBM              -> no eviction pressure, tiering
                                                    is unnecessary (recompute==tier);
  - if HBM_budget < working_set <= a fast tier  -> offload the reused pool to that
                                                    tier: the eviction cliff is
                                                    ERASED (near-100% recall, big
                                                    TTFT/throughput win);
  - if working_set exceeds every tier           -> the tier thrashes (recall rate
                                                    collapses) -> reduce the
                                                    footprint (fp8 KV) / working set
                                                    or enlarge the fastest tier.

This is a CLOSED-FORM predictor calibrated by `memos calibrate` (tier capacities
+ bandwidths) and `model_profile` (KV bytes/token, weight bytes) -- no
discrete-event simulator required. It validates cleanly against every §15 point
(see `validate_against_tier_results`).

The HBM-KV budget model, validated on §15:
    hbm_kv_budget_tokens = (util * HBM_total_bytes - weight_bytes) / kv_bytes_per_token
GB300 tray = 4 x 277.5GB = 1110GB; at util=0.5, 70B fp16 (weights 141GB) ->
414GB KV -> 1.26M tokens, i.e. the cliff sits between §15's n=128 (1.05M) and
n=192 (1.57M) -- exactly as measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memos.types import HardwareConfig

# Leave headroom on a host tier for the OS + engine host process + pinned/NIXL
# buffers (07_12: node had 952GB, cpu:800 was rejected for OOM risk, cpu:700 ok).
_HOST_HEADROOM = 0.85


@dataclass
class TierBudget:
    """A memory tier expressed as KV-token capacity for this model."""

    name: str
    capacity_gb: float
    capacity_tokens: float
    bandwidth_gbps: float
    latency_us: float
    kind: str  # "hbm" | "host" | "remote" | "disk"


@dataclass
class Recommendation:
    """A tier-placement plan for a reused KV working set."""

    working_set_tokens: float
    working_set_gb: float
    hbm_kv_budget_tokens: float
    regime: str  # "hbm_resident" | "tiering_win" | "overflow_thrash"
    placement_tier: str | None
    recommended_offload_gb: float | None
    predicted_hit_rate: float
    verdict: str
    tier_budgets: list[TierBudget] = field(default_factory=list)
    reload_beats_recompute: bool | None = None
    notes: list[str] = field(default_factory=list)


def _classify(name: str) -> str:
    low = name.lower()
    if "hbm" in low and "peer" not in low:
        return "hbm"
    if "peer" in low:
        return "peer"
    if any(k in low for k in ("lpddr", "dram", "host", "cpu")):
        return "host"
    if any(k in low for k in ("nvme", "ssd", "disk")):
        return "disk"
    return "remote"


def tier_budgets(
    hw: HardwareConfig,
    kv_bytes_per_token: float,
    weight_bytes: float,
    util: float = 0.9,
) -> list[TierBudget]:
    """Convert a calibrated HardwareConfig into per-tier KV-token capacities.

    HBM is special: its KV budget is (util * HBM_total - weights), because
    weights are co-resident. HBM total scales by gpu_count (TP spans the tray).
    Host DRAM tiers (lpddr local+remote / any host tier) are AGGREGATED into one
    "host" pool since KVBM's cpu offload draws from all host RAM. `peer` (TP-
    shared) tiers are excluded from the KV-offload budget for now (single-replica
    TP): they hold shard state, not spare KV capacity.
    """
    if kv_bytes_per_token <= 0:
        raise ValueError("kv_bytes_per_token must be > 0")

    budgets: list[TierBudget] = []
    host_cap_gb = 0.0
    host_bw = 0.0
    host_lat = 0.0
    for t in hw.tiers:
        kind = _classify(t.name)
        if kind == "hbm":
            hbm_total_bytes = t.capacity_gb * 1e9 * max(hw.gpu_count, 1)
            usable = max(util * hbm_total_bytes - weight_bytes, 0.0)
            budgets.append(
                TierBudget(
                    name=t.name,
                    capacity_gb=t.capacity_gb * max(hw.gpu_count, 1),
                    capacity_tokens=usable / kv_bytes_per_token,
                    bandwidth_gbps=t.bandwidth_gbps,
                    latency_us=t.latency_us,
                    kind="hbm",
                )
            )
        elif kind == "host":
            host_cap_gb += t.capacity_gb
            # conservative: slowest host link governs reload
            host_bw = (
                t.bandwidth_gbps if host_bw == 0 else min(host_bw, t.bandwidth_gbps)
            )
            host_lat = max(host_lat, t.latency_us)
        elif kind == "disk":
            budgets.append(
                TierBudget(
                    name=t.name,
                    capacity_gb=t.capacity_gb,
                    capacity_tokens=t.capacity_gb * 1e9 / kv_bytes_per_token,
                    bandwidth_gbps=t.bandwidth_gbps,
                    latency_us=t.latency_us,
                    kind="disk",
                )
            )
        # peer/remote intentionally skipped (see docstring)

    if host_cap_gb > 0:
        budgets.append(
            TierBudget(
                name="host_dram",
                capacity_gb=host_cap_gb,
                capacity_tokens=host_cap_gb * 1e9 / kv_bytes_per_token,
                bandwidth_gbps=host_bw,
                latency_us=host_lat,
                kind="host",
            )
        )
    return budgets


def recommend_kv_placement(
    hw: HardwareConfig,
    kv_bytes_per_token: float,
    weight_bytes: float,
    working_set_tokens: float,
    util: float = 0.9,
    prefill_flops_per_token: float | None = None,
    peak_flops: float | None = None,
) -> Recommendation:
    """Recommend where to place a reused KV working set, per the host-capacity law.

    `working_set_tokens` is the total reused KV footprint (e.g. num_prefixes *
    prefix_len for a pool of distinct-but-reused prefixes). Returns the regime,
    the offload target tier (if any), a recommended offload size, and the
    predicted recall/hit-rate. If `prefill_flops_per_token` and `peak_flops` are
    given, also checks whether reload-from-tier beats recompute at the bandwidth
    level (the sign that can flip on slow-host platforms).
    """
    budgets = tier_budgets(hw, kv_bytes_per_token, weight_bytes, util)
    hbm = next((b for b in budgets if b.kind == "hbm"), None)
    if hbm is None:
        raise ValueError("no HBM tier found in hardware config")
    hbm_budget = hbm.capacity_tokens
    ws_gb = working_set_tokens * kv_bytes_per_token / 1e9

    # Offload candidates: everything except HBM, fastest first.
    offload = sorted(
        (b for b in budgets if b.kind != "hbm"),
        key=lambda b: b.bandwidth_gbps,
        reverse=True,
    )

    notes: list[str] = []
    reload_beats_recompute: bool | None = None
    if prefill_flops_per_token and peak_flops:
        # per reused prefix token: reload time vs recompute time.
        recompute_s = prefill_flops_per_token / peak_flops
        # use the fastest offload tier's bandwidth for the check
        bw = offload[0].bandwidth_gbps if offload else 0.0
        reload_s = (kv_bytes_per_token / (bw * 1e9)) if bw > 0 else float("inf")
        reload_beats_recompute = reload_s < recompute_s
        notes.append(
            f"reload {reload_s * 1e6:.2f}us/tok vs recompute "
            f"{recompute_s * 1e6:.2f}us/tok -> "
            f"{'reload' if reload_beats_recompute else 'recompute'} cheaper"
        )

    # Regime 1: fits HBM -> no pressure.
    if working_set_tokens <= hbm_budget:
        return Recommendation(
            working_set_tokens=working_set_tokens,
            working_set_gb=ws_gb,
            hbm_kv_budget_tokens=hbm_budget,
            regime="hbm_resident",
            placement_tier=hbm.name,
            recommended_offload_gb=None,
            predicted_hit_rate=1.0,
            verdict=(
                f"Working set ({working_set_tokens / 1e6:.2f}M tok) fits the HBM KV "
                f"budget ({hbm_budget / 1e6:.2f}M tok). No tiering needed; recompute "
                "and offload perform equally (no eviction pressure)."
            ),
            tier_budgets=budgets,
            reload_beats_recompute=reload_beats_recompute,
            notes=notes,
        )

    # Regime 2: pressure exists -> find the fastest tier that HOLDS the pool.
    for tier in offload:
        if working_set_tokens <= tier.capacity_tokens * _HOST_HEADROOM:
            win = reload_beats_recompute is not False  # None or True -> assume win
            regime = "tiering_win" if win else "overflow_thrash"
            verdict = (
                f"Offload the reused pool ({ws_gb:.0f}GB) to '{tier.name}' "
                f"(cap {tier.capacity_gb:.0f}GB). Working set fits the tier -> "
                "predicted near-full recall, eviction cliff ERASED."
                if win
                else (
                    f"'{tier.name}' holds the pool but its bandwidth "
                    f"({tier.bandwidth_gbps:.0f}GB/s) makes reload SLOWER than "
                    "recompute -> tiering will not win here; prefer recompute or a "
                    "faster tier."
                )
            )
            return Recommendation(
                working_set_tokens=working_set_tokens,
                working_set_gb=ws_gb,
                hbm_kv_budget_tokens=hbm_budget,
                regime=regime,
                placement_tier=tier.name,
                recommended_offload_gb=round(ws_gb / _HOST_HEADROOM, 1),
                predicted_hit_rate=min(1.0, tier.capacity_tokens / working_set_tokens),
                verdict=verdict,
                tier_budgets=budgets,
                reload_beats_recompute=reload_beats_recompute,
                notes=notes,
            )

    # Regime 3: exceeds every tier -> thrash.
    largest = max(offload, key=lambda b: b.capacity_tokens, default=None)
    hit = min(1.0, largest.capacity_tokens / working_set_tokens) if largest else 0.0
    return Recommendation(
        working_set_tokens=working_set_tokens,
        working_set_gb=ws_gb,
        hbm_kv_budget_tokens=hbm_budget,
        regime="overflow_thrash",
        placement_tier=largest.name if largest else None,
        recommended_offload_gb=None,
        predicted_hit_rate=hit,
        verdict=(
            f"Working set ({ws_gb:.0f}GB) exceeds every tier "
            f"(largest '{largest.name}' {largest.capacity_gb:.0f}GB). "
            if largest
            else "No offload tier available. "
        )
        + "Expect recall thrash. Remedies: halve KV footprint (fp8 KV cache), "
        "shrink the reused working set, add a larger fast tier, or accept recompute.",
        tier_budgets=budgets,
        reload_beats_recompute=reload_beats_recompute,
        notes=notes,
    )


def validate_against_tier_results(
    results_path: str,
    kv_bytes_per_token: float,
    weight_bytes: float,
    hw: HardwareConfig,
    util: float = 0.5,
) -> list[dict]:
    """Compare recommender predictions to measured §15-style tier sweeps.

    For each measured (config, saturated point), predict the regime from the
    config's working set + the offload tier size encoded in its name (kvbm_cpuN),
    and check whether "predicted win" matches "measured high host hit-rate".
    Returns a list of per-config comparison rows.
    """
    import re

    from memos.serving.tier_plot import load_tier_points

    budgets = tier_budgets(hw, kv_bytes_per_token, weight_bytes, util)
    hbm_budget = next(b.capacity_tokens for b in budgets if b.kind == "hbm")

    data = load_tier_points(results_path)
    # group measured points by config; take the highest-rate (most pressure) point
    by_name: dict[str, object] = {}
    for p in data.points:
        cur = by_name.get(p.name)
        if cur is None or float(p.request_rate) > float(cur.request_rate):  # type: ignore
            by_name[p.name] = p

    rows: list[dict] = []
    for name, pt in by_name.items():
        cpu_m = re.search(r"cpu(\d+)", name.lower())
        if not cpu_m:  # only kvbm arms carry a tier size to validate the law against
            continue
        # §15-family names encode the pool as pool<N>/prefixpool<N>; prefix_len=8192.
        prefix_len = 8192
        ws_tokens = pt.num_prefixes * prefix_len  # type: ignore
        cpu_gb = int(cpu_m.group(1))
        cpu_tokens = cpu_gb * 1e9 / kv_bytes_per_token

        # The directly-measurable host-capacity law: the reused pool is recalled
        # (high host hit-rate) IFF it fits the chosen offload tier (WS <= cpu:N).
        # `has_pressure` (WS > HBM budget) additionally distinguishes a cliff-
        # erasing WIN from merely "cached but there was no cliff to beat".
        predicted_fits = ws_tokens <= cpu_tokens
        has_pressure = ws_tokens > hbm_budget
        predicted_hit_ceiling = 1.0 if predicted_fits else cpu_tokens / ws_tokens
        measured_hit = pt.hit_rate  # type: ignore
        measured_hit_high = measured_hit > 0.5

        rows.append(
            {
                "config": name,
                "cpu_gb": cpu_gb,
                "ws_Mtok": round(ws_tokens / 1e6, 2),
                "hbm_budget_Mtok": round(hbm_budget / 1e6, 2),
                "cpu_cap_Mtok": round(cpu_tokens / 1e6, 2),
                "has_pressure": has_pressure,
                "predicted_fits_tier": predicted_fits,
                "predicted_win": predicted_fits and has_pressure,
                "predicted_hit_ceiling": round(predicted_hit_ceiling, 3),
                "measured_hit": round(measured_hit, 3),
                "measured_hit_high": measured_hit_high,
                "match": predicted_fits == measured_hit_high,
            }
        )
    return rows
