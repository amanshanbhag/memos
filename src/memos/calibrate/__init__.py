"""memos.calibrate — hardware calibration package.

Submodules:
  probe     — single-node measurement (runs on each allocated node)
  assemble  — merges per-node JSONs + nccl-tests logs into final YAML
  manifest  — generates scheduler manifests (SLURM sbatch, K8s MPIJob)
  platforms — loads and validates platform config YAMLs
"""

from memos.calibrate.probe import (
    ProbeResult,
    TierResult,
    probe_node,
    probe_to_json,
)

__all__ = ["ProbeResult", "TierResult", "probe_node", "probe_to_json"]
