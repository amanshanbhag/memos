"""
memos calibrate manifest — render scheduler-specific manifests from Jinja2 templates.

Generates SLURM sbatch scripts or Kubernetes MPIJob YAMLs that orchestrate:
  1. Per-node probing (_probe-node)
  2. nccl-tests (sendrecv_perf and all_reduce_perf)
  3. Results assembly (_assemble)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import jinja2
except ImportError:
    jinja2 = None  # type: ignore[assignment]

from memos.calibrate.platforms import PlatformConfig

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "manifests"

DEFAULT_CONTAINER_IMAGE = "nvcr.io/nvidia/vllm:26.05-py3"


def _resolve_container_mounts(user_mounts: str | None, output_dir: str) -> str | None:
    """Build container mounts, ensuring output_dir and cwd are always mounted.

    If the user provides --container-mounts, use theirs as-is.
    Otherwise, auto-mount:
      1. cwd (assumes memos repo root, so memos is accessible inside container)
      2. output_dir (so probe results and assembled YAMLs land on shared storage)
    Deduplicates if output_dir is under cwd.
    """
    if user_mounts:
        return user_mounts

    import os

    cwd = os.getcwd()
    clean_out = output_dir.rstrip("/")

    mounts: list[str] = []
    mounts.append(f"{cwd}:{cwd}")
    if clean_out.startswith("/") and not clean_out.startswith(cwd):
        mounts.append(f"{clean_out}:{clean_out}")

    return ",".join(mounts) if mounts else None


def _host_to_container_path(host_path: str, container_mounts: str | None) -> str:
    """Translate a host path to its container-internal equivalent.

    Parses --container-mounts (host:container,...) and returns the
    remapped path if host_path falls under a mount source. If no mount
    matches, returns the original path (identity mount or no container).
    """
    if not container_mounts:
        return host_path

    clean = host_path.rstrip("/")
    for mount in container_mounts.split(","):
        parts = mount.split(":")
        if len(parts) < 2:
            continue
        src = parts[0].rstrip("/")
        dst = parts[1].rstrip("/")
        if clean == src:
            return dst
        if clean.startswith(src + "/"):
            return dst + clean[len(src) :]
    return host_path


def _resolve_repo_dir(container_mounts: str | None) -> str:
    """Determine where the memos repo lives inside the container."""
    import os

    return _host_to_container_path(os.getcwd(), container_mounts)


def _parse_nodelist(nodelist: str | None) -> list[str]:
    """Split a comma-separated nodelist string into individual hostnames.

    Returns an empty list if nodelist is None or empty (templates use
    {% if nodelist_items %} to conditionally render node affinity).
    """
    if not nodelist:
        return []
    return [n.strip() for n in nodelist.split(",") if n.strip()]


def _get_env() -> "jinja2.Environment":
    if jinja2 is None:
        raise ImportError(
            "jinja2 is required for manifest generation. "
            "Install with: pip install jinja2"
        )
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_calibrate_manifest(
    platform: PlatformConfig,
    scheduler: str,
    nodes: int,
    output_dir: str,
    final_output: str,
    nodelist: str | None = None,
    time: str = "00:10:00",
    account: str | None = None,
    partition: str | None = None,
    container_image: str | None = None,
    container_mounts: str | None = None,
    slurm_args: list[str] | None = None,
    namespace: str | None = None,
    pvc: str | None = None,
) -> str:
    """Render a calibration manifest (SLURM sbatch or K8s MPIJob YAML).

    The manifest orchestrates: per-node probe, nccl-tests, assembly.
    --nodelist works for both schedulers: SLURM uses #SBATCH --nodelist,
    K8s uses nodeAffinity with kubernetes.io/hostname.
    """
    env = _get_env()

    if scheduler == "slurm":
        template = env.get_template("slurm_calibrate.sbatch.j2")
    elif scheduler == "k8s":
        template = env.get_template("k8s_calibrate.yaml.j2")
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}. Use 'slurm' or 'k8s'.")

    resolved_image = container_image or DEFAULT_CONTAINER_IMAGE
    resolved_mounts = _resolve_container_mounts(container_mounts, output_dir)
    repo_dir = _resolve_repo_dir(resolved_mounts)
    c_output_dir = _host_to_container_path(output_dir, resolved_mounts).rstrip("/")
    c_final_output = _host_to_container_path(final_output, resolved_mounts)

    context: dict[str, Any] = {
        "platform": platform,
        "nodes": nodes,
        "gpus_per_node": platform.gpus_per_node,
        "host_output_dir": output_dir.rstrip("/"),
        "output_dir": c_output_dir,
        "final_output": c_final_output,
        "nccl_env": platform.nccl_env,
        "nccl_max_bytes": platform.nccl_max_bytes,
        "repo_dir": repo_dir,
        "nodelist": nodelist or "",
        "nodelist_items": _parse_nodelist(nodelist),
        "time": time,
        "account": account or "",
        "partition": partition or "",
        "image": resolved_image,
        "container_image": resolved_image,
        "container_mounts": resolved_mounts or "",
        "slurm_args": slurm_args or [],
        "k8s_namespace": namespace or "",
        "pvc": pvc,
    }

    return template.render(**context)


def render_run_manifest(
    scheduler: str,
    workload: str,
    hw_path: str,
    model: str,
    gpus_per_node: int,
    output_dir: str,
    nodes: int = 1,
    tp: int | None = None,
    input_tokens: str | None = None,
    repeats: int = 3,
    batch: int = 1,
    batch_mode: str = "static",
    max_num_seqs: int | None = None,
    cache_mode: str = "cold",
    output_tokens: str = "128",
    engine_args: list[str] | None = None,
    env_vars: list[str] | None = None,
    slurm_args: list[str] | None = None,
    nodelist: str | None = None,
    time: str = "02:00:00",
    account: str | None = None,
    partition: str | None = None,
    container_image: str | None = None,
    container_mounts: str | None = None,
    namespace: str | None = None,
    pvc: str | None = None,
) -> str:
    """Render a benchmark run manifest (SLURM sbatch or K8s Job YAML)."""
    env = _get_env()

    if scheduler == "slurm":
        template = env.get_template("slurm_run.sbatch.j2")
    elif scheduler == "k8s":
        template = env.get_template("k8s_run.yaml.j2")
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}. Use 'slurm' or 'k8s'.")

    resolved_image = container_image or DEFAULT_CONTAINER_IMAGE
    resolved_mounts = _resolve_container_mounts(container_mounts, output_dir)
    repo_dir = _resolve_repo_dir(resolved_mounts)
    c_output_dir = _host_to_container_path(output_dir, resolved_mounts).rstrip("/")
    c_hw_path = _host_to_container_path(hw_path, resolved_mounts)

    context: dict[str, Any] = {
        "workload": workload,
        "hw_path": c_hw_path,
        "model": model,
        "repo_dir": repo_dir,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "host_output_dir": output_dir.rstrip("/"),
        "output_dir": c_output_dir,
        "tp": tp,
        "input_tokens": input_tokens,
        "repeats": repeats,
        "batch": batch,
        "batch_mode": batch_mode,
        "max_num_seqs": max_num_seqs,
        "cache_mode": cache_mode,
        "output_tokens": output_tokens,
        "engine_args": engine_args or [],
        "env_vars": env_vars or [],
        "slurm_args": slurm_args or [],
        "nodelist": nodelist or "",
        "nodelist_items": _parse_nodelist(nodelist),
        "time": time,
        "account": account or "",
        "partition": partition or "",
        "image": resolved_image,
        "container_image": resolved_image,
        "container_mounts": resolved_mounts or "",
        "k8s_namespace": namespace or "",
        "pvc": pvc,
    }

    return template.render(**context)


def render_bench_serve_manifest(
    scheduler: str,
    hw_path: str,
    model: str,
    gpus_per_node: int,
    output_dir: str,
    nodes: int = 1,
    tp: int | None = None,
    isl: int = 1024,
    osl: int = 128,
    num_prompts: int = 500,
    warmup_prompts: int = 64,
    request_rate: str = "inf",
    max_concurrency: str | None = None,
    dataset: str = "random",
    dataset_path: str | None = None,
    prefix_len: int = 0,
    range_ratio: str | None = None,
    port: int = 8000,
    percentiles: str = "90,95,99",
    engine_args: list[str] | None = None,
    kv_offload: str | None = None,
    env_vars: list[str] | None = None,
    slurm_args: list[str] | None = None,
    nodelist: str | None = None,
    time: str = "02:00:00",
    account: str | None = None,
    partition: str | None = None,
    container_image: str | None = None,
    container_mounts: str | None = None,
    namespace: str | None = None,
    pvc: str | None = None,
) -> str:
    """Render an open-loop serving benchmark manifest (server+client colocated)."""
    env = _get_env()

    if scheduler == "slurm":
        template = env.get_template("slurm_bench_serve.sbatch.j2")
    elif scheduler == "k8s":
        template = env.get_template("k8s_bench_serve.yaml.j2")
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}. Use 'slurm' or 'k8s'.")

    resolved_image = container_image or DEFAULT_CONTAINER_IMAGE
    resolved_mounts = _resolve_container_mounts(container_mounts, output_dir)
    repo_dir = _resolve_repo_dir(resolved_mounts)
    c_output_dir = _host_to_container_path(output_dir, resolved_mounts).rstrip("/")
    c_hw_path = _host_to_container_path(hw_path, resolved_mounts)

    context: dict[str, Any] = {
        "hw_path": c_hw_path,
        "model": model,
        "repo_dir": repo_dir,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "host_output_dir": output_dir.rstrip("/"),
        "output_dir": c_output_dir,
        "tp": tp,
        "isl": isl,
        "osl": osl,
        "num_prompts": num_prompts,
        "warmup_prompts": warmup_prompts,
        "request_rate": request_rate,
        "max_concurrency": max_concurrency or "",
        "dataset": dataset,
        "dataset_path": dataset_path or "",
        "prefix_len": prefix_len,
        "range_ratio": range_ratio if range_ratio is not None else "",
        "port": port,
        "percentiles": percentiles,
        "engine_args": engine_args or [],
        "kv_offload": kv_offload or "",
        "env_vars": env_vars or [],
        "slurm_args": slurm_args or [],
        "nodelist": nodelist or "",
        "nodelist_items": _parse_nodelist(nodelist),
        "time": time,
        "account": account or "",
        "partition": partition or "",
        "image": resolved_image,
        "container_image": resolved_image,
        "container_mounts": resolved_mounts or "",
        "k8s_namespace": namespace or "",
        "pvc": pvc,
    }

    return template.render(**context)
