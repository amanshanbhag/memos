"""Load and validate platform configs from hardware/platforms/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PlatformConfig:
    """Minimal static facts about a GPU platform.

    Only contains what cannot be derived at runtime.
    Everything else (bandwidth, capacity, latency) is measured by the probe.
    """

    name: str
    gpu_model: str
    gpus_per_node: int
    nccl_env: dict[str, str] = field(default_factory=dict)
    nccl_max_bytes: str = "8G"


_PLATFORMS_DIR = Path(__file__).resolve().parents[3] / "hardware" / "platforms"

# Fallback: check relative to cwd (for installed packages)
if not _PLATFORMS_DIR.exists():
    _cwd_candidate = Path.cwd() / "hardware" / "platforms"
    if _cwd_candidate.exists():
        _PLATFORMS_DIR = _cwd_candidate


def load_platform(name: str, platforms_dir: Path | None = None) -> PlatformConfig:
    """Load a platform config by name.

    Args:
        name: Platform name (e.g. "gb300"). Matches the YAML filename.
        platforms_dir: Override directory containing platform YAMLs.
    """
    base = platforms_dir or _PLATFORMS_DIR
    path = base / f"{name}.yaml"
    if not path.exists():
        available = list_platforms(base)
        raise FileNotFoundError(
            f"Platform '{name}' not found at {path}. "
            f"Available: {', '.join(available) or 'none'}"
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    return PlatformConfig(
        name=raw["name"],
        gpu_model=raw["gpu_model"],
        gpus_per_node=raw["gpus_per_node"],
        nccl_env=raw.get("nccl_env") or {},
        nccl_max_bytes=raw.get("nccl_max_bytes", "8G"),
    )


def list_platforms(platforms_dir: Path | None = None) -> list[str]:
    """Return sorted list of available platform names."""
    base = platforms_dir or _PLATFORMS_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))
