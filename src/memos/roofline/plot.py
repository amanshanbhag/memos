from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RooflinePoint:
    label: str
    arithmetic_intensity: float
    measured_flops_per_sec: float


def plot_roofline(
    points: list[RooflinePoint],
    tier_bandwidths_gbps: dict[str, float],
    peak_flops: float,
    title: str = "Memory Roofline",
    output: str | Path | None = None,
) -> None:
    """Plot classic roofline: FLOPs/s vs arithmetic intensity."""
    if not points:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    ais = np.array([max(p.arithmetic_intensity, 1e-12) for p in points], dtype=float)
    yvals = np.array(
        [max(p.measured_flops_per_sec, 1e-12) for p in points], dtype=float
    )

    x_min = min(ais.min(), 1e-4)
    x_max = max(ais.max(), 1.0)
    x_line = np.logspace(np.log10(x_min / 2), np.log10(x_max * 2), 400)

    for tier_name, bw_gbps in sorted(
        tier_bandwidths_gbps.items(), key=lambda kv: kv[1], reverse=True
    ):
        y_line = np.minimum(peak_flops, x_line * bw_gbps * 1e9)
        ax.plot(x_line, y_line, label=f"{tier_name} ({bw_gbps:.1f} GB/s)")

    ax.hlines(
        y=peak_flops,
        xmin=x_line.min(),
        xmax=x_line.max(),
        colors="k",
        linestyles="--",
        label=f"Compute peak ({peak_flops:.2e} FLOP/s)",
    )

    ax.scatter(ais, yvals, c="red", marker="x", s=80, zorder=5, label="Measured")
    for point in points:
        ax.annotate(
            point.label,
            (
                max(point.arithmetic_intensity, 1e-12),
                max(point.measured_flops_per_sec, 1e-12),
            ),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (FLOP/s)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()

    plt.close(fig)
