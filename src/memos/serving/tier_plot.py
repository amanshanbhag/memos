"""Tiering comparison plotting for the KV capacity-regime experiment (sweep §14+).

The "money plot" for the KV-tiering arc (findings #13-#15): with a POOL of
distinct-but-reused prefixes (the `prefix_repetition` dataset), we sweep the
working-set size (`num_prefixes`) and compare, at each size, a `recompute`
baseline (vLLM V1 discards + recomputes an evicted prefix) against `kvbm`
(Dynamo KVBM reloads it from host RAM/SSD). Where the pool overflows HBM (and
whether it still fits the host tier) decides whether reload beats recompute.

Design decisions (learned from the 07_11 §14 analysis):
- Compare arms at the SAME offered rate. An earlier version picked each arm's
  independent max-throughput point, which silently compared recompute@rate16
  vs kvbm@rate8 -- apples to oranges. Here every (arm, rate) is its own series.
- DROP collapsed points (output_throughput == 0): under heavy overflow a run
  can complete zero requests; plotting it as "0 tok/s" is misleading, so we
  exclude it and report it separately.
- Only consider capacity-sweep configs (`num_prefixes > 0`), so non-tiering
  `serve_*` / single-prefix dirs don't pollute the x-axis at n=0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

# Rendered headless in sweeps; force Agg so it never depends on a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from memos.results import load_result  # noqa: E402

_METRIC_TO_FIELD = {
    "output_throughput": "output_throughput",
    "request_throughput": "request_throughput",
    "p99_ttft_ms": "ttft_p99",
    "median_ttft_ms": "ttft_p50",
    "p99_tpot_ms": "tpot_p99",
    "kv_cache_usage_peak": "kv_usage_peak",
    "num_preemptions": "num_preemptions",
}


@dataclass
class TierPoint:
    """One (config, offered-rate) point in the working-set sweep."""

    name: str  # config dir name
    arm: str  # "recompute" | "kvbm cpu400" | ...
    num_prefixes: int  # x-axis: working-set size (distinct prefixes)
    request_rate: str = "?"  # offered rate label
    output_throughput: float = 0.0
    request_throughput: float = 0.0
    ttft_p99: float = 0.0
    ttft_p50: float = 0.0
    tpot_p99: float = 0.0
    kv_usage_peak: float = 0.0
    num_preemptions: float = 0.0
    # KVBM tier-movement counters/gauges (names vary across versions) captured
    # generically: e.g. kvbm_onboard_blocks, kvbm_offload_blocks_d2h,
    # kvbm_*_hit_rate_peak.
    kvbm: dict[str, float] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """A point is usable only if the run actually served tokens."""
        return self.output_throughput > 0

    @property
    def onboard(self) -> float:
        """Blocks recalled HBM<-host (the reload-beats-recompute precondition)."""
        return sum(v for k, v in self.kvbm.items() if "onboard" in k)

    @property
    def offload(self) -> float:
        """Blocks evicted HBM->host (write-through volume)."""
        return sum(v for k, v in self.kvbm.items() if "offload" in k)

    @property
    def hit_rate(self) -> float:
        return max((v for k, v in self.kvbm.items() if "hit_rate" in k), default=0.0)


def _arm_of(name: str) -> str:
    """Derive a readable arm label from the config dir name."""
    low = name.lower()
    if "recompute" in low:
        return "recompute"
    m = re.search(r"kvbm[_-]([a-z0-9]+)", low)
    if m:
        return f"kvbm {m.group(1)}"
    if "kvbm" in low:
        return "kvbm"
    return "baseline"


@dataclass
class TierData:
    """All valid points plus the collapsed ones we dropped (for reporting)."""

    points: list[TierPoint] = field(default_factory=list)
    dropped: list[TierPoint] = field(default_factory=list)

    def rates(self) -> list[str]:
        return sorted(
            {p.request_rate for p in self.points},
            key=lambda r: (r == "inf", float(r) if r not in ("inf", "?") else 1e9),
        )

    def arms(self) -> list[str]:
        return sorted({p.arm for p in self.points})

    def series(self, arm: str, rate: str) -> list[TierPoint]:
        return sorted(
            (p for p in self.points if p.arm == arm and p.request_rate == rate),
            key=lambda p: p.num_prefixes,
        )


def load_tier_points(results_path: str | Path) -> TierData:
    """Load bench_serve_*.json under a path into per-(config, rate) TierPoints.

    Only capacity-sweep configs (num_prefixes > 0) are kept. Points whose run
    served zero tokens (collapsed under overflow) are separated into `dropped`.
    """
    path = Path(results_path)
    files = [path] if path.is_file() else sorted(path.rglob("bench_serve_*.json"))

    data = TierData()
    for rf in files:
        result = load_result(rf)
        if result.workload != "bench_serve":
            continue
        name = rf.parent.name
        num_prefixes = int(result.params.get("num_prefixes", 0) or 0)
        if num_prefixes == 0:  # fall back to the dir name (e.g. prefixpool192)
            m = re.search(r"(?:prefixpool|pool|prefixes?)(\d+)", name.lower())
            if m:
                num_prefixes = int(m.group(1))
        if num_prefixes == 0:  # not a capacity-sweep config -> skip entirely
            continue

        by_rate: dict[str, dict[str, float]] = {}
        kvbm_by_rate: dict[str, dict[str, float]] = {}
        for mtr in result.metrics:
            rate = str(mtr.context.get("request_rate", "?"))
            by_rate.setdefault(rate, {})[mtr.name] = mtr.value
            if mtr.name.startswith("kvbm"):
                kvbm_by_rate.setdefault(rate, {})[mtr.name] = mtr.value

        for rate, vals in by_rate.items():
            pt = TierPoint(
                name=name,
                arm=_arm_of(name),
                num_prefixes=num_prefixes,
                request_rate=rate,
                kvbm=kvbm_by_rate.get(rate, {}),
            )
            for metric, fieldname in _METRIC_TO_FIELD.items():
                if metric in vals:
                    setattr(pt, fieldname, vals[metric])
            (data.points if pt.valid else data.dropped).append(pt)
    return data


def plot_tiering(
    data: TierData,
    title: str = "KV tiering vs working-set size",
    output: str | Path | None = None,
) -> None:
    """Three-panel working-set sweep: throughput, TTFT, and tiering mechanism.

    One series per (arm, offered-rate) so arms are only ever compared at the
    SAME rate (color = arm, linestyle = rate). Panel 1: output tok/s vs
    num_prefixes. Panel 2: p99 TTFT (log y). Panel 3: mechanism -- KVBM onboard
    blocks (bars, at the highest rate) and recompute preemptions (line).
    """
    if not data.points:
        return

    arms = data.arms()
    rates = data.rates()
    cmap = plt.get_cmap("tab10")
    color = {a: cmap(i % 10) for i, a in enumerate(arms)}
    styles = ["-", "--", ":", "-."]
    style = {r: styles[i % len(styles)] for i, r in enumerate(rates)}

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    ax_tp, ax_ttft, ax_mech = axes

    for arm in arms:
        for rate in rates:
            pts = data.series(arm, rate)
            if not pts:
                continue
            x = [p.num_prefixes for p in pts]
            c, ls = color[arm], style[rate]
            lbl = f"{arm} @{rate}"
            ax_tp.plot(
                x,
                [p.output_throughput for p in pts],
                ls,
                marker="o",
                color=c,
                label=lbl,
                ms=5,
            )
            ax_ttft.plot(
                x, [p.ttft_p99 for p in pts], ls, marker="o", color=c, label=lbl, ms=5
            )

    ax_tp.set_ylabel("Output throughput (tok/s)")
    ax_tp.set_title(title)
    ax_tp.grid(True, ls=":", alpha=0.3)
    ax_tp.legend(fontsize=7, framealpha=0.9)

    ax_ttft.set_yscale("log")
    ax_ttft.set_ylabel("p99 TTFT (ms)")
    ax_ttft.grid(True, which="both", ls=":", alpha=0.3)
    ax_ttft.legend(fontsize=7, framealpha=0.9)

    # Panel 3: mechanism at the highest offered rate (most pressure).
    top_rate = rates[-1] if rates else None
    kvbm_arms = [a for a in arms if "kvbm" in a]
    xs = sorted({p.num_prefixes for p in data.points})
    if kvbm_arms and xs and top_rate is not None:
        span = (xs[1] - xs[0]) if len(xs) > 1 else max(xs[0], 1)
        width = 0.8 * span / max(len(kvbm_arms), 1)
        for j, arm in enumerate(kvbm_arms):
            pts = {p.num_prefixes: p for p in data.series(arm, top_rate)}
            heights = [pts[x].onboard if x in pts else 0.0 for x in xs]
            offs = [x + (j - (len(kvbm_arms) - 1) / 2) * width for x in xs]
            ax_mech.bar(
                offs,
                heights,
                width=width,
                color=color[arm],
                alpha=0.7,
                label=f"{arm} onboard @{top_rate}",
            )
    ax_mech.set_ylabel(f"KVBM onboard blocks @{top_rate}")
    ax_mech.set_xlabel("Distinct prefixes in working set (num_prefixes)")
    ax_mech.grid(True, ls=":", alpha=0.3)

    ax_pre = ax_mech.twinx()
    for arm in arms:
        if "recompute" not in arm or top_rate is None:
            continue
        pts = data.series(arm, top_rate)
        if pts:
            ax_pre.plot(
                [p.num_prefixes for p in pts],
                [p.num_preemptions for p in pts],
                "--s",
                color=color[arm],
                label=f"{arm} preemptions @{top_rate}",
                ms=5,
            )
    ax_pre.set_ylabel("recompute preemptions")

    h1, l1 = ax_mech.get_legend_handles_labels()
    h2, l2 = ax_pre.get_legend_handles_labels()
    if h1 or h2:
        ax_mech.legend(h1 + h2, l1 + l2, fontsize=7, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()
    plt.close(fig)
