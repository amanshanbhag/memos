from __future__ import annotations

from pathlib import Path

import yaml

from memos.types import HardwareConfig, TierSpec


def load_hardware(path: str | Path) -> HardwareConfig:
    """Load and validate a hardware config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hardware config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    tiers = [
        TierSpec(
            name=t["name"],
            capacity_gb=t["capacity_gb"],
            bandwidth_gbps=t["bandwidth_gbps"],
            latency_us=t["latency_us"],
            cost_per_gb_hour=t.get("cost_per_gb_hour", 0.0),
            numa_distance=t.get("numa_distance", 10),
            scope=t.get("scope", "node"),
            multiplicity=t.get("multiplicity", 1),
            method=t.get("method", ""),
        )
        for t in raw.get("tiers", [])
    ]

    return HardwareConfig(
        name=raw["name"],
        gpu=raw["gpu"],
        gpu_count=raw.get("gpu_count", 1),
        tiers=tiers,
        metadata=raw.get("metadata", {}),
    )
