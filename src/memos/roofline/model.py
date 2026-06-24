from __future__ import annotations

from dataclasses import dataclass

from memos.types import HardwareConfig


@dataclass
class RooflineCeilings:
    """The three ceilings that bound achievable tokens/sec."""

    compute_ceiling: float  # tokens/sec limited by peak FLOPS
    bandwidth_ceiling: float  # tokens/sec limited by memory bandwidth
    fault_ceiling: float  # tokens/sec limited by tier-miss stalls
    bottleneck: str  # "compute", "bandwidth", or "fault"


def compute_ceilings(
    hw: HardwareConfig,
    flops_per_token: float,
    bytes_per_token: float,
    fault_rate: float = 0.0,
    fault_latency_us: float = 0.0,
    peak_flops: float | None = None,
) -> RooflineCeilings:
    """Calculate roofline ceilings for a workload on given hardware.

    Args:
        hw: Hardware configuration with tier specs.
        flops_per_token: Compute cost per output token (FLOPS).
        bytes_per_token: Measured data movement per token (bytes).
        fault_rate: Tier misses per token (0.0 = no faults).
        fault_latency_us: Average latency per tier miss (microseconds).
        peak_flops: Override peak FLOPS (otherwise pulled from hw.metadata).
    """
    if peak_flops is None:
        peak_flops = hw.metadata.get("peak_flops", 0.0)

    # Compute ceiling: how fast can we generate if compute is the only limit?
    compute = peak_flops / flops_per_token if flops_per_token > 0 else float("inf")

    # Bandwidth ceiling: how fast if memory bandwidth is the only limit?
    # Uses the primary tier (HBM) bandwidth
    hbm_bw = hw.tiers[0].bandwidth_gbps * 1e9 if hw.tiers else float("inf")
    bandwidth = hbm_bw / bytes_per_token if bytes_per_token > 0 else float("inf")

    # Fault ceiling: how fast if tier-miss stalls are the only limit?
    if fault_rate > 0 and fault_latency_us > 0:
        fault = 1.0 / (fault_rate * fault_latency_us * 1e-6)
    else:
        fault = float("inf")

    effective = min(compute, bandwidth, fault)
    if effective == compute:
        bottleneck = "compute"
    elif effective == bandwidth:
        bottleneck = "bandwidth"
    else:
        bottleneck = "fault"

    return RooflineCeilings(
        compute_ceiling=compute,
        bandwidth_ceiling=bandwidth,
        fault_ceiling=fault,
        bottleneck=bottleneck,
    )
