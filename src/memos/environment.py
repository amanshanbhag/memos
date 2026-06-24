from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Environment:
    """Captured software/hardware environment for reproducibility."""

    driver_version: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_count: int = 0
    python_version: str = ""
    os: str = ""
    arch: str = ""
    packages: dict[str, str] = field(default_factory=dict)  # name -> version


def detect_environment() -> Environment:
    """Detect the current software and hardware environment."""
    env = Environment(
        python_version=platform.python_version(),
        os=platform.system(),
        arch=platform.machine(),
    )

    # GPU info from nvidia-smi
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name,count",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split("\n")[0]
        )
        parts = [p.strip() for p in out.split(",")]
        env.driver_version = parts[0]
        env.gpu_name = parts[1]
        env.gpu_count = int(parts[2])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # CUDA version from nvcc
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        for line in out.split("\n"):
            if "release" in line:
                env.cuda_version = line.split("release")[-1].split(",")[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Key package versions
    for pkg in ["vllm", "torch", "nccl", "numpy", "triton"]:
        try:
            mod = __import__(pkg)
            env.packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # NCCL version (often not directly importable)
    try:
        import torch

        env.packages["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass

    return env
