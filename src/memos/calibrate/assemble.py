"""
Assemble per-node probe JSONs and NCCL logs into a final measured hardware YAML.

Aggregation strategy:
  - Per-node tiers (HBM, peer, host DRAM, NVMe): median across N nodes
  - Cross-node tier (remote_hbm): peak bus BW from nccl-tests at largest message size
  - Outlier nodes flagged if >10% deviation from median for any tier
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from memos.calibrate.platforms import PlatformConfig


def assemble(
    results_dir: str | Path,
    platform: PlatformConfig,
    output: str | Path | None = None,
) -> str:
    """Merge per-node probe results and NCCL logs into a final hardware YAML.

    Args:
        results_dir: Directory containing node_*.json and nccl_*.log files.
        platform: Platform config for metadata.
        output: If provided, write YAML to this path.

    Returns:
        The assembled YAML as a string.
    """
    results_dir = Path(results_dir)
    node_files = sorted(results_dir.glob("node_*.json"))
    if not node_files:
        raise FileNotFoundError(f"No node_*.json files found in {results_dir}")

    node_results = []
    for nf in node_files:
        with open(nf) as f:
            node_results.append(json.load(f))

    num_nodes = len(node_results)
    gpu_name = node_results[0].get("gpu_name", "unknown")
    gpus_per_node = node_results[0].get("gpu_count", platform.gpus_per_node)
    total_gpus = num_nodes * gpus_per_node

    # Collect hostnames
    hostnames = [nr.get("hostname", f"node_{i}") for i, nr in enumerate(node_results)]

    # Collect software versions from first node (should be identical across nodes)
    software = node_results[0].get("software", {})

    # Aggregate per-node tiers using median
    aggregated_tiers = _aggregate_tiers(node_results, gpus_per_node)

    # Parse NCCL logs for cross-node tier
    remote_tier = _parse_nccl_logs(results_dir, num_nodes, gpus_per_node)
    if remote_tier:
        aggregated_tiers.append(remote_tier)

    # Build output YAML
    # gpu_count is per-node (matches HardwareConfig and hand-crafted YAMLs)
    doc: dict[str, Any] = {
        "name": f"{platform.name}_measured",
        "gpu": gpu_name,
        "gpu_count": gpus_per_node,
        "tiers": aggregated_tiers,
        "metadata": {
            "platform": platform.name,
            "calibrated_at": datetime.now().isoformat(),
            "calibration_nodes": num_nodes,
            "calibration_total_gpus": total_gpus,
            "nodelist": hostnames,
            "software": software,
            "aggregation": _build_aggregation_metadata(node_results),
        },
    }

    # Add NCCL metadata if allreduce log exists
    allreduce_bw = _parse_nccl_max_busbw(results_dir / "nccl_allreduce.log")
    if allreduce_bw > 0:
        doc["metadata"]["nccl"] = {
            "allreduce_busbw_gbps": round(allreduce_bw, 1),
        }
    sendrecv_bw = _parse_nccl_max_busbw(results_dir / "nccl_sendrecv.log")
    if sendrecv_bw > 0:
        doc["metadata"].setdefault("nccl", {})["sendrecv_busbw_gbps"] = round(
            sendrecv_bw, 1
        )

    # Inject per-GPU dense tensor-core peak FLOPS from the platform config so the
    # roofline compute ceiling has spec data (calibration measures bandwidth, not
    # FLOPS). Written as flat metadata keys (peak_flops_fp16, ...) that the
    # roofline reads, plus a generic `peak_flops` default (fp16).
    if platform.peak_flops:
        for precision, value in platform.peak_flops.items():
            doc["metadata"][f"peak_flops_{precision}"] = value
        default_flops = platform.peak_flops.get("fp16") or platform.peak_flops.get(
            "bf16"
        )
        if default_flops:
            doc["metadata"]["peak_flops"] = default_flops

    yaml_str = yaml.dump(doc, default_flow_style=False, sort_keys=False)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_str)

    return yaml_str


def _aggregate_tiers(
    node_results: list[dict[str, Any]], gpus_per_node: int
) -> list[dict[str, Any]]:
    """Compute median bandwidth/latency/capacity across nodes per tier name."""
    tier_data: dict[str, list[dict[str, Any]]] = {}
    for nr in node_results:
        for tier in nr.get("tiers", []):
            tier_data.setdefault(tier["name"], []).append(tier)

    aggregated = []
    for tier_name, samples in tier_data.items():
        bw_values = [s["bandwidth_gbps"] for s in samples]
        lat_values = [s["latency_us"] for s in samples]
        cap_values = [s["capacity_gb"] for s in samples]

        median_bw = round(statistics.median(bw_values), 1)
        median_lat = round(statistics.median(lat_values), 3)
        median_cap = round(statistics.median(cap_values), 1)

        first = samples[0]
        aggregated.append(
            {
                "name": tier_name,
                "scope": first.get("scope", "node"),
                "capacity_gb": median_cap,
                "bandwidth_gbps": median_bw,
                "latency_us": median_lat,
                "multiplicity": first.get("multiplicity", 1),
                "method": first.get("method", ""),
            }
        )

    return aggregated


def _build_aggregation_metadata(
    node_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build min/max ranges and outlier detection per tier."""
    OUTLIER_THRESHOLD = 0.10

    tier_data: dict[str, list[tuple[str, float]]] = {}
    for nr in node_results:
        hostname = nr.get("hostname", "unknown")
        for tier in nr.get("tiers", []):
            tier_data.setdefault(tier["name"], []).append(
                (hostname, tier["bandwidth_gbps"])
            )

    meta: dict[str, Any] = {"method": "median"}
    outlier_nodes: list[str] = []

    for tier_name, host_bw_pairs in tier_data.items():
        bw_values = [bw for _, bw in host_bw_pairs]
        median_bw = statistics.median(bw_values)
        meta[f"{tier_name}_bw_range"] = [
            round(min(bw_values), 1),
            round(max(bw_values), 1),
        ]

        if median_bw > 0:
            for hostname, bw in host_bw_pairs:
                deviation = abs(bw - median_bw) / median_bw
                if deviation > OUTLIER_THRESHOLD and hostname not in outlier_nodes:
                    outlier_nodes.append(hostname)

    meta["outlier_nodes"] = outlier_nodes
    return meta


def _parse_nccl_logs(
    results_dir: Path, num_nodes: int, gpus_per_node: int
) -> dict[str, Any] | None:
    """Parse nccl_sendrecv.log to extract cross-node bandwidth."""
    sendrecv_log = results_dir / "nccl_sendrecv.log"
    if not sendrecv_log.exists():
        return None

    max_busbw = _parse_nccl_max_busbw(sendrecv_log)
    if max_busbw <= 0:
        return None

    # Latency: smallest message size busbw
    min_lat = _parse_nccl_min_latency(sendrecv_log)

    total_remote_gpus = (num_nodes - 1) * gpus_per_node

    return {
        "name": "remote_hbm",
        "scope": "domain",
        "capacity_gb": 0.0,  # filled by user or computed from node count
        "bandwidth_gbps": round(max_busbw, 1),
        "latency_us": round(min_lat, 1) if min_lat > 0 else 3.0,
        "multiplicity": total_remote_gpus,
        "method": "nccl-tests:sendrecv_perf:max_busbw",
    }


def _parse_nccl_max_busbw(log_path: Path) -> float:
    """Extract the maximum bus bandwidth from an nccl-tests log."""
    if not log_path.exists():
        return 0.0

    max_bw = 0.0
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # nccl-tests output columns: size count type redop root time algbw busbw ...
        parts = line.split()
        if len(parts) >= 8:
            try:
                busbw = float(parts[-1])
                max_bw = max(max_bw, busbw)
            except ValueError:
                continue
    return max_bw


def _parse_nccl_min_latency(log_path: Path) -> float:
    """Extract latency from smallest message size in nccl-tests log.

    At small message sizes, time is dominated by latency.
    Returns microseconds.
    """
    if not log_path.exists():
        return 0.0

    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 6:
            try:
                # First data line is smallest message size; column index 5 is time (us)
                return float(parts[5])
            except (ValueError, IndexError):
                continue
    return 0.0
