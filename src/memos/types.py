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
class InferenceConfig:
    """Inference runtime knobs that affect roofline calculations."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    batch_size: int = 1
    cache_mode: str = "cold"
    weight_dtype_bytes: float = 2.0
    kv_dtype_bytes: float = 2.0
    weight_group_size: int = 0  # 0 means no group quantization metadata overhead
    activation_dtype: str = "fp16"
    speculative: bool = False
    mtp: bool = False
    engine_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete result from running one workload on one hardware config."""

    workload: str
    hardware: str
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    params: dict[str, Any] = field(default_factory=dict)
    metrics: list[MetricSample] = field(default_factory=list)
    inference_config: InferenceConfig | None = None
    environment: dict[str, Any] = field(
        default_factory=dict
    )  # from detect_environment()
    raw: dict[str, Any] = field(default_factory=dict)
