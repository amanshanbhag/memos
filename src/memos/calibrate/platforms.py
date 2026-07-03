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
    # Per-GPU DENSE tensor-core peak throughput by precision (FLOPS/OPS).
    # Populated from the yaml `peak_flops` block if present, else from the
    # cited per-GPU-model table below. Consumed by the roofline compute ceiling.
    peak_flops: dict[str, float] = field(default_factory=dict)


# Per-GPU DENSE tensor-core peak throughput (FLOPS), WITHOUT structured sparsity.
# Rationale: LLM inference GEMMs are dense, so the dense peak is the correct
# roofline compute ceiling. NVIDIA marketing figures are usually quoted WITH
# 2:4 sparsity; dense is half of those (datasheets state "1/2 lower without
# sparsity"). Values are per single GPU and are therefore identical across the
# _Ngpu platform variants (a10g, a10g_4gpu, ...). Keys match _infer_precision()
# in the CLI (fp16/bf16/fp8/int8/fp4). Units: raw FLOPS (e.g. 989e12 = 989 TF).
_PEAK_FLOPS_BY_GPU: dict[str, dict[str, float]] = {
    # AWS A10G (GA102, Ampere). AWS-specific variant, lower-clocked than the A10.
    # AWS A10G datasheet: FP16/BF16 TC 70 TF dense (140 TF*); INT8 140 TOPS dense.
    # No FP8 on Ampere. Src: d1.awsstatic.com A10G datasheet (2022-02-17).
    "A10G": {"fp16": 70e12, "bf16": 70e12, "int8": 140e12},
    # A100 (GA100, Ampere) SXM/PCIe. FP16/BF16 TC 312 TF dense; INT8 624 TOPS dense.
    # No FP8 on Ampere. Src: NVIDIA A100 datasheet (nvidia-a100-datasheet.pdf).
    "A100": {"fp16": 312e12, "bf16": 312e12, "int8": 624e12},
    # H100 SXM (GH100, Hopper). Dense = half of datasheet sparse figures:
    # FP16/BF16 989 TF, FP8 1979 TF, INT8 1979 TOPS. Src: NVIDIA H100 datasheet
    # ("* Shown with sparsity. Specifications 1/2 lower without sparsity.").
    "H100": {"fp16": 989e12, "bf16": 989e12, "fp8": 1979e12, "int8": 1979e12},
    # H200 (GH100, Hopper): identical compute to H100 SXM (only memory differs).
    "H200": {"fp16": 989e12, "bf16": 989e12, "fp8": 1979e12, "int8": 1979e12},
    # L4 (AD104, Ada Lovelace). Dense: FP16/BF16 121 TF, FP8 242.5 TF, INT8 242.5 TOPS.
    # Src: NVIDIA L4 datasheet / Lenovo lp1717 ("one-half lower without sparsity").
    "L4": {"fp16": 121e12, "bf16": 121e12, "fp8": 242.5e12, "int8": 242.5e12},
    # L40S (AD102, Ada Lovelace). Dense: FP16/BF16 362 TF, FP8 733 TF, INT8 733 TOPS.
    # Src: NVIDIA/PNY L40S datasheet & Lenovo lp1812 (362.05 | 733* form).
    "L40S": {"fp16": 362e12, "bf16": 362e12, "fp8": 733e12, "int8": 733e12},
    # B200 (Blackwell). Dense = half of datasheet sparse: FP16/BF16 2.25 PF,
    # FP8 4.5 PF, FP4 9 PF, INT8 4.5 POPS. Src: NVIDIA Blackwell datasheet
    # ("All Tensor Core numbers except FP64 with sparsity").
    "B200": {
        "fp16": 2250e12,
        "bf16": 2250e12,
        "fp8": 4500e12,
        "fp4": 9000e12,
        "int8": 4500e12,
    },
    # GB200 Grace-Blackwell superchip: per-GPU compute equals B200.
    "GB200": {
        "fp16": 2250e12,
        "bf16": 2250e12,
        "fp8": 4500e12,
        "fp4": 9000e12,
        "int8": 4500e12,
    },
    # B300 (Blackwell Ultra). Anchor: NVIDIA states GB300 NVL72 = 1.1 EF DENSE FP4
    # over 72 GPUs => ~15.3 PF/GPU dense FP4 (developer.nvidia.com "Inside NVIDIA
    # Blackwell Ultra"; GB300 superchip = 30 PF dense NVFP4 / 2 GPUs = 15 PF/GPU).
    # FP8/FP16 derived via the standard 4:2:1 tensor ratio: FP8 7.5 PF, FP16 3.75 PF.
    # CAVEAT: some third-party trackers list B300 FP16 2.5 / FP8 5 PF; the
    # NVIDIA-anchored FP4 + standard ratio is used here. Revisit on official B300 sheet.
    "B300": {
        "fp16": 3750e12,
        "bf16": 3750e12,
        "fp8": 7500e12,
        "fp4": 15000e12,
        "int8": 7500e12,
    },
    # GB300 Grace-Blackwell Ultra superchip: per-GPU compute equals B300.
    "GB300": {
        "fp16": 3750e12,
        "bf16": 3750e12,
        "fp8": 7500e12,
        "fp4": 15000e12,
        "int8": 7500e12,
    },
    # RTX PRO 6000 Blackwell Server Edition (GB202). NVIDIA product page (sparse):
    # FP4 4 PF, FP8 2 PF, FP16/BF16 1 PF => dense (half): FP16 500 TF, FP8 1000 TF,
    # FP4 2000 TF, INT8 1000 TOPS. Src: nvidia.com rtx-pro-6000-blackwell-server-edition.
    "RTX_PRO_6000": {
        "fp16": 500e12,
        "bf16": 500e12,
        "fp8": 1000e12,
        "fp4": 2000e12,
        "int8": 1000e12,
    },
}


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

    gpu_model = raw["gpu_model"]
    raw_flops = raw.get("peak_flops")
    if raw_flops:
        peak_flops = {str(k): float(v) for k, v in raw_flops.items()}
    else:
        peak_flops = dict(_PEAK_FLOPS_BY_GPU.get(gpu_model, {}))

    return PlatformConfig(
        name=raw["name"],
        gpu_model=gpu_model,
        gpus_per_node=raw["gpus_per_node"],
        nccl_env=raw.get("nccl_env") or {},
        nccl_max_bytes=raw.get("nccl_max_bytes", "8G"),
        peak_flops=peak_flops,
    )


def list_platforms(platforms_dir: Path | None = None) -> list[str]:
    """Return sorted list of available platform names."""
    base = platforms_dir or _PLATFORMS_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))
