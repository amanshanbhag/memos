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
    # Gate 2 (feasibility) -- see assess_kv_feasibility / finding #23.
    churn: float | None = None
    feasibility_risk: str | None = None  # "safe" | "moderate" | "high"
    feasible: bool | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gate 2: FEASIBILITY (research notes finding #23).
#
# Gate 1 (reload-vs-recompute + host-capacity) answers "is offload CHEAPER?".
# Gate 2 answers "is offload DEPLOYABLE at this load?" -- because the KVBM
# offload path can CRASH (vLLM scheduler.py:647 assertion) under sustained
# offload/onboard CHURN, independent of whether it would be cheaper.
#
# Empirical basis (measured tierhost llama70b sweeps; churn = working_set /
# HBM_KV_budget, using each GPU's MEASURED KV pool):
#   H100  (80GB, util0.9, 444k tok): churn 2.36 CRASH@r8, 3.54 CRASH@r8,
#                                     4.72 CRASH@r12   -> early failure ~churn 2.4
#   GB300 (util0.27,      487k tok): churn 2.15 ok,   3.23 ok,  4.31 CRASH@r12
#   GB300/GB200 (util0.5, >=721k):   churn <=1.6      -> always ok
# Two regimes emerge:
#   * a PORTABLE ceiling ~4: crashes on ANY HW at high concurrency;
#   * an EARLY-FAILURE floor ~2: crashes only on small-HBM/slow-host parts.
# Concurrency matters: at rate<8 even H100 churn 4.72 survived; crashes appear
# at rate>=8.
# ISOLATED (finding #24, B200 07_17 vs GB200): the factor that lowers the
# early-failure threshold is the HOST INTERCONNECT (first order), with silicon
# generation second order. Matched Blackwell silicon + matched churn, flipping only
# the host link: GB200 (C2C) rode churn ~3.0 at all rates; B200 (PCIe) degraded 3.3x
# at churn ~2.3 and crashed at churn ~3.0. H100 (PCIe + Hopper) crashed at churn ~2.3.
# So the per-platform ceiling is keyed on (host_link, silicon) -- see
# `safe_churn_ceiling_for` / `_CEILING_BY_HOST` below.
# ---------------------------------------------------------------------------

# Below this churn there is no meaningful offload traffic (pool ~fits HBM).
_CHURN_NO_TRAFFIC = 1.0
# Concurrency (offered rate / max in-flight) at/above which the race becomes fatal.
_RISK_CONCURRENCY = 8.0
# Churn above which the offload path crashes on EVERY GPU measured (portable).
_CHURN_PORTABLE_CEILING = 4.0

# Calibrated gate-2 churn ceilings by (host_link, silicon) -- finding #24.
# The feasibility ceiling is set FIRST-ORDER by the host link (C2C >> PCIe at
# matched churn/silicon); silicon generation is a second-order downgrade
# (Hopper < Blackwell on PCIe). Measured: C2C Blackwell safe to ~3.2 (-> 4.0);
# PCIe Blackwell (B200) degrades ~2.3 / crashes ~3.0 (-> 2.5); PCIe Hopper (H100)
# crashes ~2.3 (-> 2.0). C2C Hopper unmeasured -> assume the link dominates.
_CEILING_BY_HOST: dict[tuple[str, str], float] = {
    ("c2c", "blackwell"): 4.0,
    ("c2c", "hopper"): 4.0,
    ("pcie", "blackwell"): 2.5,
    ("pcie", "hopper"): 2.0,
}


def safe_churn_ceiling_for(host_link: str, silicon: str = "blackwell") -> float:
    """Calibrated gate-2 churn ceiling for a platform (finding #24).

    host_link: 'c2c' (Grace NVLink-C2C) or 'pcie'; silicon: 'blackwell' | 'hopper'.
    Falls back to the conservative 2.0 (small-HBM / slow-host) for unknown combos.
    """
    return _CEILING_BY_HOST.get((host_link.lower(), silicon.lower()), 2.0)


def assess_kv_feasibility(
    churn: float,
    concurrency: float | None,
    safe_churn_ceiling: float = 2.0,
) -> tuple[str, bool, str]:
    """Gate 2: will the KVBM offload path stay alive at this churn + concurrency?

    Returns (risk_level, feasible, note). `churn` = working_set / HBM_KV_budget.
    `safe_churn_ceiling` is the per-platform churn below which offload is safe at
    high concurrency; default 2.0 is the conservative (small-HBM/slow-host, i.e.
    H100-like) value. Large-HBM / fast-host (C2C) platforms tolerate more -- pass
    a higher ceiling (Blackwell C2C measured safe to ~3.2). Above the PORTABLE
    ceiling (~4) offload crashes on every GPU measured.
    """
    if concurrency is None:
        concurrency = _RISK_CONCURRENCY  # assume load unless told otherwise

    if churn <= _CHURN_NO_TRAFFIC:
        return (
            "safe",
            True,
            (
                f"churn {churn:.2f} <= 1: pool ~fits HBM, negligible offload traffic; "
                "no feasibility risk."
            ),
        )
    if concurrency < _RISK_CONCURRENCY:
        return (
            "safe",
            True,
            (
                f"churn {churn:.2f} at low concurrency ({concurrency:g} < "
                f"{_RISK_CONCURRENCY:g}): offload traffic tolerable; the race did not "
                "fire below rate 8 in any measured run."
            ),
        )
    if churn >= _CHURN_PORTABLE_CEILING:
        return (
            "high",
            False,
            (
                f"churn {churn:.2f} >= portable ceiling {_CHURN_PORTABLE_CEILING:g} at "
                f"concurrency {concurrency:g}: offload path crashes on EVERY GPU "
                "measured (scheduler.py:647). Do NOT offload; recompute or shrink the "
                "working set / add HBM."
            ),
        )
    if churn > safe_churn_ceiling:
        return (
            "moderate",
            False,
            (
                f"churn {churn:.2f} in the platform-dependent band "
                f"({safe_churn_ceiling:g}..{_CHURN_PORTABLE_CEILING:g}] at concurrency "
                f"{concurrency:g}: fails on small-HBM/slow-host (e.g. H100) but survives "
                "on large-HBM/fast-host (C2C Blackwell). Feasibility is HW-dependent; "
                "treat as risky unless this platform's ceiling is known higher."
            ),
        )
    return (
        "safe",
        True,
        (
            f"churn {churn:.2f} <= platform ceiling {safe_churn_ceiling:g}: offload "
            "path stayed alive in all measured runs at this churn."
        ),
    )


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
    concurrency: float | None = None,
    safe_churn_ceiling: float = 2.0,
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

    # Gate 2 (feasibility): offload/onboard churn = working set / HBM KV budget.
    churn = working_set_tokens / hbm_budget if hbm_budget > 0 else float("inf")
    feas_risk, feasible, feas_note = assess_kv_feasibility(
        churn, concurrency, safe_churn_ceiling
    )

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
            churn=churn,
            feasibility_risk=feas_risk,
            feasible=feasible,
            notes=notes + [feas_note],
        )

    # Regime 2: pressure exists -> find the fastest tier that HOLDS the pool.
    for tier in offload:
        if working_set_tokens <= tier.capacity_tokens * _HOST_HEADROOM:
            cost_win = reload_beats_recompute is not False  # None/True -> cost win
            # Gate 1 says "cheaper"; gate 2 must also say "deployable".
            win = cost_win and feasible
            regime = "tiering_win" if win else "overflow_thrash"
            if not cost_win:
                verdict = (
                    f"'{tier.name}' holds the pool but its bandwidth "
                    f"({tier.bandwidth_gbps:.0f}GB/s) makes reload SLOWER than "
                    "recompute -> tiering will not win here; prefer recompute or a "
                    "faster tier."
                )
            elif not feasible:
                verdict = (
                    f"'{tier.name}' holds the pool AND reload is cheaper than "
                    f"recompute, BUT gate 2 fails: {feas_note} Offload would be "
                    "cheaper-if-it-ran, but the path is not deployable at this "
                    "load; prefer recompute (or reduce concurrency / working set / "
                    "add HBM)."
                )
            else:
                verdict = (
                    f"Offload the reused pool ({ws_gb:.0f}GB) to '{tier.name}' "
                    f"(cap {tier.capacity_gb:.0f}GB). Working set fits the tier, "
                    "reload beats recompute, and churn is within the feasible band "
                    "-> predicted near-full recall, eviction cliff ERASED."
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
                churn=churn,
                feasibility_risk=feas_risk,
                feasible=feasible,
                notes=notes + [feas_note],
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
        churn=churn,
        feasibility_risk=feas_risk,
        feasible=feasible,
        notes=notes + [feas_note],
    )


# ---------------------------------------------------------------------------
# MULTI-TIER placement (Track D3): evaluate BOTH gates PER TIER and pick a
# placement. The novel output vs v1 (single host tier) is the CROSSOVER TIER --
# the slowest tier where reload still beats recompute -- which moves UP the
# hierarchy as GPU FLOPS grow. So the optimal placement tier is hardware-
# dependent: e.g. NVMe offload can PAY on Hopper and LOSE on Blackwell for the
# same model, because faster compute makes recompute cheaper.
#
#   reload_time/tok    = kv_bytes_per_token / tier_bandwidth
#   recompute_time/tok = prefill_flops_per_token / peak_flops   (2*P_active/FLOPS)
#   crossover bw*      = kv_bytes_per_token * peak_flops / prefill_flops_per_token
# ---------------------------------------------------------------------------


@dataclass
class TierDecision:
    """Per-tier evaluation of the two gates for a reused KV working set."""

    name: str
    kind: str  # "hbm" | "host" | "disk" | "remote"
    bandwidth_gbps: float
    capacity_gb: float
    capacity_tokens: float
    fits: bool  # capacity gate: does the working set fit (with headroom)?
    reload_us_per_tok: float
    recompute_us_per_tok: float
    reload_wins: bool  # gate 1 (cost): reload cheaper than recompute?
    feasible: bool  # gate 2 (feasibility): offload path survives churn+concurrency?
    usable: bool  # fits AND reload_wins AND feasible


@dataclass
class MultiTierPlacement:
    """A hardware-calibrated per-tier placement plan (Track D3)."""

    working_set_tokens: float
    working_set_gb: float
    hbm_kv_budget_tokens: float
    churn: float
    recompute_us_per_tok: float
    tiers: list[TierDecision]
    recommended_tier: str  # tier name | "hbm" (fits) | "recompute" (no tier usable)
    crossover_tier: str | None  # slowest tier where reload still beats recompute
    verdict: str


def place_kv_multitier(
    hw: HardwareConfig,
    kv_bytes_per_token: float,
    weight_bytes: float,
    working_set_tokens: float,
    prefill_flops_per_token: float,
    peak_flops: float,
    util: float = 0.9,
    concurrency: float | None = None,
    safe_churn_ceiling: float = 2.0,
) -> MultiTierPlacement:
    """Recommend a KV placement tier by evaluating both gates on EVERY tier.

    Unlike `recommend_kv_placement` (single aggregated host tier, v1), this keeps
    each non-HBM tier (host / disk / remote) separate, computes the reload-vs-
    recompute crossover per tier, and returns (a) the recommended tier = the
    FASTEST tier that fits the pool AND beats recompute AND is feasible, and (b)
    the CROSSOVER tier = the SLOWEST tier where reload still beats recompute (the
    hardware-dependent boundary that is this work's multi-tier contribution).
    """
    if peak_flops <= 0 or prefill_flops_per_token <= 0:
        raise ValueError("peak_flops and prefill_flops_per_token must be > 0")

    # HBM budget (weights co-resident, spans TP group).
    hbm_tier = next((t for t in hw.tiers if _classify(t.name) == "hbm"), None)
    if hbm_tier is None:
        raise ValueError("no HBM tier found in hardware config")
    hbm_total = hbm_tier.capacity_gb * 1e9 * max(hw.gpu_count, 1)
    hbm_budget = max(util * hbm_total - weight_bytes, 0.0) / kv_bytes_per_token
    churn = working_set_tokens / hbm_budget if hbm_budget > 0 else float("inf")
    ws_gb = working_set_tokens * kv_bytes_per_token / 1e9

    recompute_us = prefill_flops_per_token / peak_flops * 1e6
    _, feasible, _ = assess_kv_feasibility(churn, concurrency, safe_churn_ceiling)

    decisions: list[TierDecision] = []
    for t in hw.tiers:
        kind = _classify(t.name)
        if kind in ("hbm", "peer"):
            continue  # HBM is the source, peer holds shard state (not spare KV)
        cap_tok = t.capacity_gb * 1e9 / kv_bytes_per_token
        fits = working_set_tokens <= cap_tok * _HOST_HEADROOM
        reload_us = (
            kv_bytes_per_token / (t.bandwidth_gbps * 1e9) * 1e6
            if t.bandwidth_gbps > 0
            else float("inf")
        )
        reload_wins = reload_us < recompute_us
        decisions.append(
            TierDecision(
                name=t.name,
                kind=kind,
                bandwidth_gbps=t.bandwidth_gbps,
                capacity_gb=t.capacity_gb,
                capacity_tokens=cap_tok,
                fits=fits,
                reload_us_per_tok=reload_us,
                recompute_us_per_tok=recompute_us,
                reload_wins=reload_wins,
                feasible=feasible,
                usable=(fits and reload_wins and feasible),
            )
        )
    decisions.sort(key=lambda d: d.bandwidth_gbps, reverse=True)

    crossover = next(
        (
            d.name
            for d in sorted(decisions, key=lambda d: d.bandwidth_gbps)
            if d.reload_wins
        ),
        None,
    )

    if working_set_tokens <= hbm_budget:
        rec, verdict = "hbm", (
            f"Working set ({working_set_tokens/1e6:.2f}M tok) fits HBM "
            f"({hbm_budget/1e6:.2f}M) -> no offload needed."
        )
    else:
        usable = [d for d in decisions if d.usable]
        if usable:
            best = usable[0]  # fastest usable tier
            rec, verdict = best.name, (
                f"Offload to '{best.name}' ({best.bandwidth_gbps:.0f} GB/s): fits, "
                f"reload {best.reload_us_per_tok:.2f} < recompute {recompute_us:.2f} "
                f"us/tok, churn {churn:.2f} feasible. Slowest tier that still beats "
                f"recompute = '{crossover}'."
            )
        else:
            rec, verdict = "recompute", (
                f"No tier is usable (churn {churn:.2f}, feasible={feasible}). "
                + (
                    "All holding tiers are slower than the recompute crossover "
                    f"({kv_bytes_per_token*peak_flops/prefill_flops_per_token/1e9:.1f} "
                    "GB/s) "
                    if crossover is None
                    else f"Cheapest winning tier '{crossover}' fails the fit/feasibility "
                    "gate "
                )
                + "-> recompute (do not offload)."
            )

    return MultiTierPlacement(
        working_set_tokens=working_set_tokens,
        working_set_gb=ws_gb,
        hbm_kv_budget_tokens=hbm_budget,
        churn=churn,
        recompute_us_per_tok=recompute_us,
        tiers=decisions,
        recommended_tier=rec,
        crossover_tier=crossover,
        verdict=verdict,
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
