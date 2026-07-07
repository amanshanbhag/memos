"""SLO-curve plotting + knee extraction for `memos bench-serve` results.

The serving analogue of `roofline/plot.py`: instead of positioning points on a
compute/bandwidth roofline, it plots latency-vs-achieved-load curves (the
classic serving figure) so the SLO knee -- the load past which tail latency
blows up -- is visible per config.

We use ACHIEVED request throughput (req/s) as the x-axis rather than the
offered request-rate, because the offered rate includes `inf` (unthrottled),
which has no finite position; achieved throughput naturally places the
saturation point at the right edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

# Serving plots are saved to file (in sweeps/headless nodes); force the
# non-interactive Agg backend so rendering never depends on a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from memos.results import load_result  # noqa: E402


@dataclass
class ServingPoint:
    """One offered-load point (one `vllm bench serve` run) for a config."""

    request_rate: str  # offered rate label ("8", "inf", ...)
    request_throughput: float = 0.0  # achieved req/s (x-axis)
    output_throughput: float = 0.0  # tok/s
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    tpot_p50: float = 0.0
    tpot_p99: float = 0.0
    kv_usage_peak: float = 0.0
    cpu_usage_peak: float = 0.0
    running_peak: float = 0.0
    waiting_peak: float = 0.0
    num_preemptions: float = 0.0


@dataclass
class ServingSeries:
    """All load points for one config (one result file)."""

    name: str
    points: list[ServingPoint] = field(default_factory=list)

    def sorted_points(self) -> list[ServingPoint]:
        return sorted(self.points, key=lambda p: p.request_throughput)


_METRIC_TO_FIELD = {
    "request_throughput": "request_throughput",
    "output_throughput": "output_throughput",
    "median_ttft_ms": "ttft_p50",
    "p99_ttft_ms": "ttft_p99",
    "median_tpot_ms": "tpot_p50",
    "p99_tpot_ms": "tpot_p99",
    "kv_cache_usage_peak": "kv_usage_peak",
    "cpu_cache_usage_peak": "cpu_usage_peak",
    "running_batch_peak": "running_peak",
    "waiting_peak": "waiting_peak",
    "num_preemptions": "num_preemptions",
}


def load_serving_series(results_path: str | Path) -> list[ServingSeries]:
    """Load bench_serve_*.json under a path, one ServingSeries per config dir."""
    path = Path(results_path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("bench_serve_*.json"))

    series: list[ServingSeries] = []
    for rf in files:
        result = load_result(rf)
        if result.workload != "bench_serve":
            continue
        # One file may contain multiple configs only in theory; key by dir name.
        name = rf.parent.name
        by_rate: dict[str, ServingPoint] = {}
        for m in result.metrics:
            rate = str(m.context.get("request_rate", "?"))
            pt = by_rate.setdefault(rate, ServingPoint(request_rate=rate))
            field_name = _METRIC_TO_FIELD.get(m.name)
            if field_name is not None:
                setattr(pt, field_name, m.value)
        if by_rate:
            series.append(ServingSeries(name=name, points=list(by_rate.values())))
    return series


def find_knee(
    series: ServingSeries, ttft_slo_ms: float, tpot_slo_ms: float
) -> ServingPoint | None:
    """Highest-throughput point that still meets both p99 SLOs."""
    ok = [
        p
        for p in series.sorted_points()
        if p.ttft_p99 <= ttft_slo_ms and p.tpot_p99 <= tpot_slo_ms
    ]
    return ok[-1] if ok else None


def plot_serving(
    series: list[ServingSeries],
    title: str = "Serving SLO curves",
    output: str | Path | None = None,
    ttft_slo_ms: float = 500.0,
    tpot_slo_ms: float = 50.0,
) -> None:
    """Three-panel latency/pressure vs achieved-throughput figure.

    Panel 1: p99/p50 TTFT (log y).  Panel 2: p99/p50 TPOT.  Panel 3: KV usage +
    preemptions. SLO thresholds drawn as horizontal reference lines.
    """
    if not series:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    ax_ttft, ax_tpot, ax_press = axes
    cmap = plt.get_cmap("tab10")

    for i, s in enumerate(series):
        pts = s.sorted_points()
        x = [p.request_throughput for p in pts]
        color = cmap(i % 10)

        ax_ttft.plot(
            x, [p.ttft_p99 for p in pts], "-o", color=color, label=s.name, ms=4
        )
        ax_ttft.plot(x, [p.ttft_p50 for p in pts], "--", color=color, alpha=0.4, ms=3)
        ax_tpot.plot(
            x, [p.tpot_p99 for p in pts], "-o", color=color, label=s.name, ms=4
        )
        ax_tpot.plot(x, [p.tpot_p50 for p in pts], "--", color=color, alpha=0.4, ms=3)
        ax_press.plot(
            x,
            [p.kv_usage_peak * 100 for p in pts],
            "-o",
            color=color,
            label=f"{s.name} kv%",
            ms=4,
        )
        # Mark preemption onset with an X where preemptions > 0.
        pre_x = [p.request_throughput for p in pts if p.num_preemptions > 0]
        pre_y = [p.kv_usage_peak * 100 for p in pts if p.num_preemptions > 0]
        if pre_x:
            ax_press.scatter(pre_x, pre_y, marker="X", s=90, color=color, zorder=5)

    ax_ttft.axhline(
        ttft_slo_ms, color="red", ls=":", alpha=0.6, label=f"TTFT SLO {ttft_slo_ms:g}ms"
    )
    ax_ttft.set_yscale("log")
    ax_ttft.set_ylabel("TTFT (ms)  [p99 solid, p50 dashed]")
    ax_ttft.set_title(title)
    ax_ttft.grid(True, which="both", ls=":", alpha=0.3)
    ax_ttft.legend(fontsize=7, framealpha=0.9)

    ax_tpot.axhline(
        tpot_slo_ms, color="red", ls=":", alpha=0.6, label=f"TPOT SLO {tpot_slo_ms:g}ms"
    )
    ax_tpot.set_ylabel("TPOT (ms)  [p99 solid, p50 dashed]")
    ax_tpot.grid(True, which="both", ls=":", alpha=0.3)
    ax_tpot.legend(fontsize=7, framealpha=0.9)

    ax_press.set_ylabel("KV cache usage (%)  [X = preemptions>0]")
    ax_press.set_xlabel("Achieved request throughput (req/s)")
    ax_press.grid(True, which="both", ls=":", alpha=0.3)
    ax_press.legend(fontsize=7, framealpha=0.9)

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()
    plt.close(fig)
