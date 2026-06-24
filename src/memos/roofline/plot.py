from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from memos.roofline.model import RooflineCeilings


def plot_roofline(
    ceilings: list[RooflineCeilings],
    labels: list[str],
    measured_tokens_per_sec: list[float] | None = None,
    title: str = "Memory Roofline",
    output: str | Path | None = None,
) -> None:
    """Plot a memory roofline chart.

    Args:
        ceilings: Roofline ceilings for each workload/config.
        labels: Label for each data point.
        measured_tokens_per_sec: Actual measured throughput (optional overlay).
        title: Chart title.
        output: Save path (if None, calls plt.show()).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.25

    compute_vals = [c.compute_ceiling for c in ceilings]
    bw_vals = [c.bandwidth_ceiling for c in ceilings]
    fault_vals = [
        min(c.fault_ceiling, max(compute_vals + bw_vals) * 2) for c in ceilings
    ]

    ax.bar(x - width, compute_vals, width, label="Compute ceiling")
    ax.bar(x, bw_vals, width, label="Bandwidth ceiling")
    ax.bar(x + width, fault_vals, width, label="Fault ceiling")

    if measured_tokens_per_sec:
        ax.scatter(
            x,
            measured_tokens_per_sec,
            color="red",
            zorder=5,
            s=100,
            marker="x",
            label="Measured",
        )

    ax.set_xlabel("Workload")
    ax.set_ylabel("Tokens/sec")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()

    plt.close(fig)
