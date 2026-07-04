from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RooflinePoint:
    label: str
    arithmetic_intensity: float
    measured_flops_per_sec: float
    series: str | None = None

    def series_key(self) -> str:
        if self.series:
            return self.series
        # Default: the config name prefix of "config:ISLxOSL" labels.
        return self.label.split(":", 1)[0] if ":" in self.label else "measured"


def plot_roofline(
    points: list[RooflinePoint],
    tier_bandwidths_gbps: dict[str, float],
    peak_flops: float,
    title: str = "Memory Roofline",
    output: str | Path | None = None,
) -> None:
    """Plot a classic roofline: performance (FLOP/s) vs arithmetic intensity.

    Points are colored by series (model/config) with a legend instead of
    per-point text labels, and the axes are zoomed to the measured data so the
    cluster of points is readable rather than collapsed against the origin.
    """
    if not points:
        return

    fig, ax = plt.subplots(figsize=(11, 7.5))

    groups: dict[str, list[RooflinePoint]] = defaultdict(list)
    for p in points:
        groups[p.series_key()].append(p)

    ais_all = [max(p.arithmetic_intensity, 1e-12) for p in points]
    y_all = [max(p.measured_flops_per_sec, 1e-12) for p in points]
    x_lo, x_hi = min(ais_all), max(ais_all)
    y_lo, y_hi = min(y_all), max(y_all)

    # x range: pad the measured cluster; do not force it back to ~0 (the old
    # min(ais.min(), 1e-4) collapsed every point into the far right corner).
    x_line = np.logspace(np.log10(x_lo / 3.0), np.log10(x_hi * 3.0), 400)

    # Bandwidth-ceiling diagonals (slope = BW), capped at the compute peak.
    for tier_name, bw_gbps in sorted(
        tier_bandwidths_gbps.items(), key=lambda kv: kv[1], reverse=True
    ):
        cap = peak_flops if peak_flops > 0 else np.inf
        y_line = np.minimum(cap, x_line * bw_gbps * 1e9)
        ax.plot(
            x_line, y_line, lw=1.3, alpha=0.9, label=f"{tier_name} ({bw_gbps:.0f} GB/s)"
        )

    if peak_flops > 0:
        ax.hlines(
            y=peak_flops,
            xmin=x_line.min(),
            xmax=x_line.max(),
            colors="k",
            linestyles="--",
            lw=1.2,
            label=f"compute peak ({peak_flops:.2e} FLOP/s)",
        )

    # Measured points, colored per series (model/config).
    cmap = plt.get_cmap("tab20" if len(groups) > 10 else "tab10")
    for i, series in enumerate(sorted(groups)):
        pts = groups[series]
        xs = [max(p.arithmetic_intensity, 1e-12) for p in pts]
        ys = [max(p.measured_flops_per_sec, 1e-12) for p in pts]
        ax.scatter(
            xs,
            ys,
            s=38,
            marker="o",
            color=cmap(i % cmap.N),
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
            label=series,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_line.min(), x_line.max())
    y_top = max(peak_flops, y_hi) if peak_flops > 0 else y_hi
    ax.set_ylim(y_lo / 5.0, y_top * 3.0)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (FLOP/s)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()

    plt.close(fig)
