"""Tiering comparison plotting for the KV capacity-regime experiment (sweep §14).

The "money plot" for the KV-tiering arc (findings #13-#14): with a POOL of
distinct-but-reused prefixes (the `prefix_repetition` dataset), we sweep the
working-set size (`num_prefixes`) and compare, at each size, a `recompute`
baseline (vLLM V1 discards + recomputes an evicted prefix) against `kvbm`
(Dynamo KVBM reloads it from host RAM/SSD). Where the pool overflows HBM, the
question is whether reload beats recompute.

Unlike `serving/plot.py` (latency vs offered load, one curve per config), this
positions the SATURATED operating point of each config against the working-set
size on the x-axis, one line per arm, so the crossover -- if any -- is the
figure. A third panel shows the tiering MECHANISM (KVBM host recall vs recompute
preemptions) so a null result ("reload never engaged") is legible, not silent.
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
    """One config's SATURATED operating point in the working-set sweep."""

    name: str  # config dir name
    arm: str  # "recompute" | "kvbm cpu400" | ...
    num_prefixes: int  # x-axis: working-set size (distinct prefixes)
    request_rate: str = "?"  # offered rate at the chosen point
    output_throughput: float = 0.0
    request_throughput: float = 0.0
    ttft_p99: float = 0.0
    ttft_p50: float = 0.0
    tpot_p99: float = 0.0
    kv_usage_peak: float = 0.0
    num_preemptions: float = 0.0
    # KVBM tier-movement counters/gauges (names vary across versions) captured
    # generically so the mechanism panel is robust: e.g. kvbm_onboard_blocks,
    # kvbm_offload_blocks_d2h, kvbm_*_hit_rate_peak.
    kvbm: dict[str, float] = field(default_factory=dict)

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


def load_tier_points(results_path: str | Path) -> list[TierPoint]:
    """Load bench_serve_*.json under a path, one saturated TierPoint per config.

    The saturated point is the offered-load point with the highest achieved
    output throughput -- the operating point where memory pressure (and thus any
    tiering effect) is greatest.
    """
    path = Path(results_path)
    files = [path] if path.is_file() else sorted(path.rglob("bench_serve_*.json"))

    points: list[TierPoint] = []
    for rf in files:
        result = load_result(rf)
        if result.workload != "bench_serve":
            continue
        name = rf.parent.name
        num_prefixes = int(result.params.get("num_prefixes", 0) or 0)
        if num_prefixes == 0:  # fall back to the dir name (e.g. prefixpool192)
            m = re.search(r"(?:prefixpool|prefixes?)(\d+)", name.lower())
            if m:
                num_prefixes = int(m.group(1))

        # Bucket every metric by offered rate, then pick the saturated rate.
        by_rate: dict[str, dict[str, float]] = {}
        kvbm_by_rate: dict[str, dict[str, float]] = {}
        for mtr in result.metrics:
            rate = str(mtr.context.get("request_rate", "?"))
            by_rate.setdefault(rate, {})[mtr.name] = mtr.value
            if mtr.name.startswith("kvbm"):
                kvbm_by_rate.setdefault(rate, {})[mtr.name] = mtr.value
        if not by_rate:
            continue

        best_rate = max(by_rate, key=lambda r: by_rate[r].get("output_throughput", 0.0))
        vals = by_rate[best_rate]
        pt = TierPoint(
            name=name,
            arm=_arm_of(name),
            num_prefixes=num_prefixes,
            request_rate=best_rate,
            kvbm=kvbm_by_rate.get(best_rate, {}),
        )
        for metric, fieldname in _METRIC_TO_FIELD.items():
            if metric in vals:
                setattr(pt, fieldname, vals[metric])
        points.append(pt)
    return points


def plot_tiering(
    points: list[TierPoint],
    title: str = "KV tiering vs working-set size",
    output: str | Path | None = None,
) -> None:
    """Three-panel working-set sweep: throughput, TTFT, and tiering mechanism.

    Panel 1: output tok/s vs num_prefixes, one line per arm (the crossover/win).
    Panel 2: p99 TTFT (log y) vs num_prefixes, one line per arm.
    Panel 3: mechanism -- KVBM onboard blocks (bars, left y) and recompute
    preemptions (line, right y) vs num_prefixes, so engagement (or its absence)
    is explicit.
    """
    if not points:
        return

    arms = sorted({p.arm for p in points})
    cmap = plt.get_cmap("tab10")
    color = {a: cmap(i % 10) for i, a in enumerate(arms)}

    def line(arm: str) -> list[TierPoint]:
        return sorted((p for p in points if p.arm == arm), key=lambda p: p.num_prefixes)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    ax_tp, ax_ttft, ax_mech = axes

    for arm in arms:
        pts = line(arm)
        x = [p.num_prefixes for p in pts]
        c = color[arm]
        ax_tp.plot(
            x, [p.output_throughput for p in pts], "-o", color=c, label=arm, ms=5
        )
        ax_ttft.plot(x, [p.ttft_p99 for p in pts], "-o", color=c, label=arm, ms=5)

    ax_tp.set_ylabel("Output throughput (tok/s)")
    ax_tp.set_title(title)
    ax_tp.grid(True, ls=":", alpha=0.3)
    ax_tp.legend(fontsize=8, framealpha=0.9)

    ax_ttft.set_yscale("log")
    ax_ttft.set_ylabel("p99 TTFT (ms)")
    ax_ttft.grid(True, which="both", ls=":", alpha=0.3)
    ax_ttft.legend(fontsize=8, framealpha=0.9)

    # Panel 3: mechanism. Onboard blocks (recall) as bars for kvbm arms;
    # preemptions as a line (right axis) for the recompute arm.
    kvbm_arms = [a for a in arms if "kvbm" in a]
    xs = sorted({p.num_prefixes for p in points})
    if kvbm_arms and xs:
        width = (
            0.8
            * (min(xs) if len(xs) == 1 else (xs[1] - xs[0]))
            / max(len(kvbm_arms), 1)
        )
        for j, arm in enumerate(kvbm_arms):
            pts = {p.num_prefixes: p for p in line(arm)}
            heights = [pts[x].onboard if x in pts else 0.0 for x in xs]
            offs = [x + (j - (len(kvbm_arms) - 1) / 2) * width for x in xs]
            ax_mech.bar(
                offs,
                heights,
                width=width,
                color=color[arm],
                alpha=0.7,
                label=f"{arm} onboard blocks",
            )
    ax_mech.set_ylabel("KVBM onboard blocks (recall)")
    ax_mech.set_xlabel("Distinct prefixes in working set (num_prefixes)")
    ax_mech.grid(True, ls=":", alpha=0.3)

    ax_pre = ax_mech.twinx()
    for arm in arms:
        if "recompute" not in arm:
            continue
        pts = line(arm)
        ax_pre.plot(
            [p.num_prefixes for p in pts],
            [p.num_preemptions for p in pts],
            "--s",
            color=color[arm],
            label=f"{arm} preemptions",
            ms=5,
        )
    ax_pre.set_ylabel("recompute preemptions")

    h1, l1 = ax_mech.get_legend_handles_labels()
    h2, l2 = ax_pre.get_legend_handles_labels()
    if h1 or h2:
        ax_mech.legend(h1 + h2, l1 + l2, fontsize=8, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()
    plt.close(fig)
