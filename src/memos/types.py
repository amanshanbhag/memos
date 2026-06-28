from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TierSpec:
    """A single tier in the memory hierarchy (HBM, DRAM, NVMe, etc.)."""

    name: str
    capacity_gb: float
    bandwidth_gbps: float
    latency_us: float
    cost_per_gb_hour: float = 0.0
    numa_distance: int = 10  # default to self-distance
    scope: str = "node"  # "device" | "node" | "rack" | "cluster"
    multiplicity: int = 1  # how many instances accessible
    method: str = ""  # benchmark that produced these numbers


@dataclass
class HardwareConfig:
    """Complete memory hierarchy description for a hardware platform."""

    name: str  # "gb300", "h100_sxm", etc.
    gpu: str  # "B300", "H100", etc.
    gpu_count: int = 1
    tiers: list[TierSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSample:
    """A single metric measurement at a point in time."""

    name: str  # "bytes_per_token", "stall_time_per_token", etc.
    value: float
    unit: str  # "bytes", "us", "ratio", etc.
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete result from running one workload on one hardware config."""

    workload: str
    hardware: str
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    params: dict[str, Any] = field(default_factory=dict)
    metrics: list[MetricSample] = field(default_factory=list)
    environment: dict[str, Any] = field(
        default_factory=dict
    )  # from detect_environment()
    raw: dict[str, Any] = field(default_factory=dict)
