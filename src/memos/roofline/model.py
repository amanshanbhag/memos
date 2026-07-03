from __future__ import annotations

from dataclasses import dataclass, field

from memos.types import HardwareConfig


@dataclass
class RooflineCeilings:
    """The three ceilings that bound achievable tokens/sec."""

    compute_ceiling: float  # tokens/sec limited by peak FLOPS
    bandwidth_ceiling: float  # tokens/sec limited by memory bandwidth
    fault_ceiling: float  # tokens/sec limited by tier-miss stalls
    bottleneck: str  # "compute", "bandwidth", or "fault"
    bandwidth_ceilings: dict[str, float] = field(default_factory=dict)
    selected_tier: str = ""
    effective_bandwidth_gbps: float = 0.0


def compute_ceilings(
    hw: HardwareConfig,
    flops_per_token: float,
    bytes_per_token: float,
    working_set_bytes: float | None = None,
    fault_rate: float = 0.0,
    fault_latency_us: float = 0.0,
    precision: str = "fp16",
    peak_flops: float | None = None,
) -> RooflineCeilings:
    """Calculate roofline ceilings for a workload on given hardware.

    Args:
        hw: Hardware configuration with tier specs.
        flops_per_token: Compute cost per output token (FLOPS).
        bytes_per_token: Measured data movement per token (bytes).
        fault_rate: Tier misses per token (0.0 = no faults).
        fault_latency_us: Average latency per tier miss (microseconds).
        working_set_bytes: Active footprint in bytes for spill estimation.
        precision: Compute precision selector (fp16/fp8/int8/fp4).
        peak_flops: Override peak FLOPS (otherwise pulled from hw.metadata).
    """
    if peak_flops is None:
        precision_key = f"peak_flops_{precision.lower()}"
        peak_flops = float(
            hw.metadata.get(precision_key, hw.metadata.get("peak_flops", 0.0))
        )

    # Compute ceiling: how fast can we generate if compute is the only limit?
    compute = peak_flops / flops_per_token if flops_per_token > 0 else float("inf")

    tiers = [t for t in hw.tiers if t.bandwidth_gbps > 0]
    if not tiers:
        effective_bw_gbps = float("inf")
        bandwidth = float("inf")
        per_tier: dict[str, float] = {}
        selected_tier = ""
    else:
        per_tier = {
            t.name: (
                (t.bandwidth_gbps * 1e9 / bytes_per_token)
                if bytes_per_token > 0
                else float("inf")
            )
            for t in tiers
        }
        hbm = next(
            (t for t in tiers if "hbm" in t.name.lower() and t.scope == "device"),
            tiers[0],
        )
        selected_tier = hbm.name
        effective_bw_gbps = float(hbm.bandwidth_gbps)

        if working_set_bytes and working_set_bytes > 0:
            hbm_bytes = hbm.capacity_gb * 1e9
            if working_set_bytes > hbm_bytes:
                overflow = working_set_bytes - hbm_bytes
                remainder = overflow
                weighted_inverse = hbm_bytes / (
                    working_set_bytes * max(hbm.bandwidth_gbps, 1e-12)
                )

                # Spill progressively into lower-bandwidth tiers.
                spill_tiers = sorted(
                    [t for t in tiers if t.name != hbm.name],
                    key=lambda t: t.bandwidth_gbps,
                    reverse=True,
                )
                for tier in spill_tiers:
                    capacity = max(tier.capacity_gb, 0.0) * 1e9
                    if capacity <= 0 or remainder <= 0:
                        continue
                    assigned = min(capacity, remainder)
                    weighted_inverse += assigned / (
                        working_set_bytes * max(tier.bandwidth_gbps, 1e-12)
                    )
                    remainder -= assigned
                    selected_tier = tier.name

                if remainder > 0 and spill_tiers:
                    tail = spill_tiers[-1]
                    weighted_inverse += remainder / (
                        working_set_bytes * max(tail.bandwidth_gbps, 1e-12)
                    )
                    selected_tier = tail.name

                if weighted_inverse > 0:
                    effective_bw_gbps = 1.0 / weighted_inverse

        bandwidth = (
            effective_bw_gbps * 1e9 / bytes_per_token
            if bytes_per_token > 0
            else float("inf")
        )

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
        bandwidth_ceilings=per_tier,
        selected_tier=selected_tier,
        effective_bandwidth_gbps=effective_bw_gbps,
    )
