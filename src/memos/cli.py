from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from memos.hardware import load_hardware


@click.group()
def main() -> None:
    """memos - memory roofline and benchmarks for AI inference."""
    pass


# ---------------------------------------------------------------------------
# memos run
# ---------------------------------------------------------------------------


@main.command()
@click.argument("workload_name")
@click.option("--hw", required=True, type=click.Path(), help="Hardware config YAML")
@click.option("--model", required=True, help="Model name or path")
@click.option(
    "--output", "-o", default="results/", type=click.Path(), help="Output directory"
)
@click.option(
    "--tp", default=None, type=int, help="Tensor parallel size (default: hw.gpu_count)"
)
@click.option("--pp", default=None, type=int, help="Pipeline parallel size")
@click.option("--dp", default=None, type=int, help="Data parallel size")
@click.option("--cache-mode", default="cold", type=click.Choice(["cold", "warm"]))
@click.option(
    "--input-tokens",
    "--isl",
    default=None,
    help="Input sequence lengths, comma-separated (e.g. 4096,8192,16384)",
)
@click.option("--repeats", default=3, type=int, help="Repeats per context length")
@click.option(
    "--output-tokens",
    "--osl",
    default="128",
    help="Output sequence lengths, comma-separated (e.g. 128 or 128,512,2048)",
)
@click.option(
    "--engine-arg",
    multiple=True,
    help="Extra engine args as key=value (e.g. --engine-arg max_model_len=4096)",
)
@click.option(
    "--scheduler",
    default=None,
    type=click.Choice(["slurm", "k8s"]),
    help="Generate a scheduler manifest instead of running locally",
)
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(),
    help="Write manifest to this path (default: stdout)",
)
@click.option("--nodes", default=1, type=int, help="Nodes for submission mode")
@click.option("--time", "time_limit", default="02:00:00", help="Wall time (SLURM)")
@click.option("--account", "-A", default=None, help="SLURM account")
@click.option("--partition", default=None, help="SLURM partition")
@click.option(
    "--container-image",
    default=None,
    help="Container image (default: nvcr.io/nvidia/vllm:26.05-py3)",
)
@click.option("--container-mounts", default=None, help="Container bind mounts (SLURM)")
@click.option(
    "--nodelist",
    default=None,
    help="Comma-separated node names (SLURM --nodelist / K8s nodeAffinity)",
)
@click.option("--namespace", default=None, help="K8s namespace")
@click.option("--pvc", default=None, help="K8s PVC for shared results storage")
@click.option(
    "--env",
    multiple=True,
    help="Environment variables for container (e.g. --env HF_TOKEN=hf_...)",
)
@click.option(
    "--slurm-arg",
    multiple=True,
    help="Extra SLURM directives (e.g. --slurm-arg reservation=my_res)",
)
def run(
    workload_name: str,
    hw: str,
    model: str,
    output: str,
    tp: int | None,
    pp: int | None,
    dp: int | None,
    cache_mode: str,
    input_tokens: str | None,
    repeats: int,
    output_tokens: str,
    engine_arg: tuple[str, ...],
    scheduler: str | None,
    manifest_path: str | None,
    nodes: int,
    time_limit: str,
    account: str | None,
    partition: str | None,
    container_image: str | None,
    container_mounts: str | None,
    nodelist: str | None,
    namespace: str | None,
    pvc: str | None,
    env: tuple[str, ...],
    slurm_arg: tuple[str, ...],
) -> None:
    """Run a benchmark workload (or generate a scheduler manifest with --scheduler)."""
    hw_config = load_hardware(hw)

    # --- Submission mode: generate manifest and exit ---
    if scheduler:
        from memos.calibrate.manifest import render_run_manifest

        manifest_str = render_run_manifest(
            scheduler=scheduler,
            workload=workload_name,
            hw_path=hw,
            model=model,
            gpus_per_node=hw_config.gpu_count,
            output_dir=output,
            nodes=nodes,
            tp=tp,
            input_tokens=input_tokens,
            repeats=repeats,
            cache_mode=cache_mode,
            output_tokens=output_tokens,
            engine_args=list(engine_arg),
            env_vars=list(env),
            slurm_args=list(slurm_arg),
            nodelist=nodelist,
            time=time_limit,
            account=account,
            partition=partition,
            container_image=container_image,
            container_mounts=container_mounts,
            namespace=namespace,
            pvc=pvc,
        )

        if manifest_path:
            out_path = Path(manifest_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(manifest_str)
            click.echo(f"Manifest written to {out_path}")
            if scheduler == "slurm":
                click.echo(f"Submit with: sbatch {out_path}")
            else:
                click.echo(f"Submit with: kubectl apply -f {out_path}")
        else:
            click.echo(manifest_str)
        return

    # --- Local execution mode ---
    from memos.environment import detect_environment
    from memos.metrics.throughput import ThroughputCollector
    from memos.model_profile import from_pretrained
    from memos.results import save_result
    from memos.runners.vllm_runner import VLLMRunner
    from memos.workloads.context_sweep import ContextSweep

    workloads = {
        "context_sweep": ContextSweep,
    }

    if workload_name not in workloads:
        raise click.BadParameter(
            f"Unknown workload: {workload_name}. Available: {', '.join(workloads)}"
        )

    click.echo(f"Workload:  {workload_name}")
    click.echo(f"Hardware:  {hw_config.name}")
    click.echo(f"Model:     {model}")
    click.echo(f"Cache:     {cache_mode}")
    click.echo()

    click.echo("Profiling model architecture...")
    profile = from_pretrained(model)
    click.echo(
        f"  layers={profile.num_layers} h={profile.hidden_size} "
        f"heads={profile.num_heads} kv_heads={profile.num_kv_heads}"
    )
    click.echo(f"  weight_bytes={profile.weight_bytes() / 1e9:.2f} GB")
    click.echo()

    isls = None
    if input_tokens:
        isls = [int(x.strip()) for x in input_tokens.split(",")]

    env = detect_environment()
    click.echo(
        f"GPU: {env.gpu_name} | Driver: {env.driver_version} | CUDA: {env.cuda_version}"
    )
    click.echo(f"Packages: {env.packages}")
    click.echo()

    runner = VLLMRunner()
    runner_kwargs: dict = {}
    if tp is not None:
        runner_kwargs["tensor_parallel_size"] = tp
    if pp is not None:
        runner_kwargs["pipeline_parallel_size"] = pp
    if dp is not None:
        runner_kwargs["data_parallel_size"] = dp
    runner_kwargs["enable_prefix_caching"] = cache_mode == "warm"

    for arg in engine_arg:
        k, _, v = arg.partition("=")
        if v.lower() in ("true", "false"):
            runner_kwargs[k] = v.lower() == "true"
        else:
            try:
                runner_kwargs[k] = int(v)
            except ValueError:
                try:
                    runner_kwargs[k] = float(v)
                except ValueError:
                    runner_kwargs[k] = v

    runner.setup(model, hw_config, **runner_kwargs)

    collectors = [ThroughputCollector()]

    osls = [int(x.strip()) for x in output_tokens.split(",")]
    workload = workloads[workload_name](
        isls=isls,
        osls=osls,
        repeats=repeats,
        cache_mode=cache_mode,
    )
    click.echo(f"Running: {workload.description()}")
    click.echo()

    import dataclasses

    result = workload.run(runner, hw_config, collectors, model=model)
    result.environment = dataclasses.asdict(env)
    result.raw["model_profile"] = profile.to_dict()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{workload_name}_{hw_config.name}_{timestamp}.json"
    out_path = save_result(result, Path(output) / filename)
    click.echo(f"\nResults saved to {out_path}")

    tps_samples = [m for m in result.metrics if m.name == "tokens_per_sec"]
    if tps_samples:
        by_combo: dict[tuple[int, int], list[float]] = {}
        for s in tps_samples:
            isl = s.context.get("context_length", 0)
            osl = s.context.get("output_tokens", 0)
            by_combo.setdefault((isl, osl), []).append(s.value)
        click.echo("\nTokens/sec by ISL x OSL:")
        for isl, osl in sorted(by_combo):
            vals = by_combo[(isl, osl)]
            avg = sum(vals) / len(vals)
            click.echo(f"  ISL={isl:>7}  OSL={osl:>5}: {avg:>10.1f} tok/s")

    runner.shutdown()


# ---------------------------------------------------------------------------
# memos roofline
# ---------------------------------------------------------------------------


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option(
    "--hw", required=True, type=click.Path(exists=True), help="Hardware config YAML"
)
@click.option("--output", default=None, type=click.Path(), help="Save plot to file")
def roofline(results_dir: str, hw: str, output: str | None) -> None:
    """Generate a memory roofline plot from benchmark results."""
    from memos.model_profile import ModelProfile
    from memos.results import load_result
    from memos.roofline.model import compute_ceilings
    from memos.roofline.plot import plot_roofline

    hw_config = load_hardware(hw)
    results_path = Path(results_dir)

    result_files = list(results_path.glob("*.json"))
    if not result_files:
        click.echo(f"No result JSON files found in {results_dir}")
        return

    click.echo(f"Found {len(result_files)} result file(s)")
    click.echo(f"Hardware: {hw_config.name}")

    all_ceilings = []
    all_labels = []
    all_measured = []

    for rf in sorted(result_files):
        result = load_result(rf)
        tps_samples = [m for m in result.metrics if m.name == "tokens_per_sec"]

        profile_dict = result.raw.get("model_profile")
        profile = ModelProfile.from_dict(profile_dict) if profile_dict else None

        by_ctx: dict[int, list[float]] = {}
        for s in tps_samples:
            ctx = s.context.get("context_length", 0)
            by_ctx.setdefault(ctx, []).append(s.value)

        for ctx in sorted(by_ctx):
            avg_tps = sum(by_ctx[ctx]) / len(by_ctx[ctx])
            dur_samples = [
                m
                for m in result.metrics
                if m.name == "duration_ms" and m.context.get("context_length") == ctx
            ]
            avg_dur = (
                sum(s.value for s in dur_samples) / len(dur_samples)
                if dur_samples
                else 1.0
            )

            if profile:
                flops = profile.flops_per_token(seq=ctx)
                bw = profile.bytes_per_token(seq=ctx)
            else:
                flops = float(hw_config.metadata.get("peak_flops", 1e15)) / 1000
                bw = ctx * 2

            ceilings = compute_ceilings(
                hw=hw_config,
                flops_per_token=flops,
                bytes_per_token=bw,
            )
            all_ceilings.append(ceilings)
            label = f"{ctx}T" if ctx < 1024 else f"{ctx // 1024}K"
            all_labels.append(label)
            all_measured.append(avg_tps)

    if not all_ceilings:
        click.echo("No throughput data found in results.")
        return

    plot_roofline(
        ceilings=all_ceilings,
        labels=all_labels,
        measured_tokens_per_sec=all_measured,
        title=f"Memory Roofline - {hw_config.name}",
        output=output,
    )
    if output:
        click.echo(f"Plot saved to {output}")
    else:
        click.echo("Plot displayed.")


# ---------------------------------------------------------------------------
# memos calibrate (manifest generator)
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--platform",
    "-p",
    required=True,
    help="Platform name (e.g. gb300, h100). See 'memos platforms'",
)
@click.option(
    "--scheduler",
    "-s",
    required=True,
    type=click.Choice(["slurm", "k8s"]),
    help="Target scheduler",
)
@click.option("--nodes", "-N", required=True, type=int, help="Number of nodes")
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Output path for results (host path, auto-translated for containers)",
)
@click.option(
    "--nodelist",
    default=None,
    help="Comma-separated node names (SLURM --nodelist / K8s nodeAffinity)",
)
@click.option("--time", "time_limit", default="00:10:00", help="Wall time (HH:MM:SS)")
@click.option("--account", "-A", default=None, help="SLURM account")
@click.option("--partition", default=None, help="SLURM partition")
@click.option(
    "--container-image",
    default=None,
    help="Container image (default: nvcr.io/nvidia/vllm:26.05-py3)",
)
@click.option("--container-mounts", default=None, help="Container bind mounts (SLURM)")
@click.option("--namespace", default=None, help="K8s namespace")
@click.option("--pvc", default=None, help="K8s PVC for shared results storage")
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(),
    help="Write manifest to path (default: stdout)",
)
@click.option(
    "--slurm-arg",
    multiple=True,
    help="Extra SLURM directives (e.g. --slurm-arg reservation=my_res)",
)
def calibrate(
    platform: str,
    scheduler: str,
    nodes: int,
    output_dir: str,
    nodelist: str | None,
    time_limit: str,
    account: str | None,
    partition: str | None,
    container_image: str | None,
    container_mounts: str | None,
    namespace: str | None,
    pvc: str | None,
    manifest_path: str | None,
    slurm_arg: tuple[str, ...],
) -> None:
    """Generate a calibration manifest for multi-node hardware measurement.

    Produces a SLURM sbatch script or K8s MPIJob YAML that orchestrates:
    per-node probing, cross-node NCCL tests, and result assembly.

    Examples:

      memos calibrate -p gb300 -s slurm -N 18 -o /scratch/calibrate -A myaccount
          --nodelist node[001-018]

      memos calibrate -p h100 -s k8s -N 4 -o /shared/calibrate --pvc calibrate-pvc
          --nodelist gpu-node-1,gpu-node-2,gpu-node-3,gpu-node-4

    Uses nvcr.io/nvidia/vllm:26.05-py3 by default. Override with --container-image.
    """
    from memos.calibrate.manifest import render_calibrate_manifest
    from memos.calibrate.platforms import load_platform

    plat = load_platform(platform)
    final_output = f"{output_dir}/{platform}.yaml"

    manifest_str = render_calibrate_manifest(
        platform=plat,
        scheduler=scheduler,
        nodes=nodes,
        output_dir=output_dir,
        final_output=final_output,
        nodelist=nodelist,
        time=time_limit,
        account=account,
        partition=partition,
        container_image=container_image,
        container_mounts=container_mounts,
        slurm_args=list(slurm_arg),
        namespace=namespace,
        pvc=pvc,
    )

    if manifest_path:
        out_path = Path(manifest_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(manifest_str)
        click.echo(f"Manifest written to {out_path}")
        if scheduler == "slurm":
            click.echo(f"Submit with: sbatch {out_path}")
        else:
            click.echo(f"Submit with: kubectl apply -f {out_path}")
    else:
        click.echo(manifest_str)


# ---------------------------------------------------------------------------
# memos platforms
# ---------------------------------------------------------------------------


@main.command(name="platforms")
def list_platforms_cmd() -> None:
    """List available platform configs."""
    from memos.calibrate.platforms import list_platforms, load_platform

    platforms = list_platforms()
    if not platforms:
        click.echo("No platform configs found in hardware/platforms/")
        return

    click.echo("Available platforms:\n")
    for name in platforms:
        plat = load_platform(name)
        nccl_keys = ", ".join(plat.nccl_env.keys()) if plat.nccl_env else "none"
        click.echo(
            f"  {name:<24} gpu={plat.gpu_model:<15} "
            f"gpus/node={plat.gpus_per_node}  nccl=[{nccl_keys}]"
        )


# ---------------------------------------------------------------------------
# memos list
# ---------------------------------------------------------------------------


@main.command(name="list")
def list_workloads() -> None:
    """List available workloads."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Available Workloads")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Status")

    table.add_row(
        "context_sweep",
        "Sweep context lengths, measure memory behavior vs scale",
        "ready",
    )
    table.add_row(
        "thrashing_sweep",
        "Ramp concurrent sessions until throughput collapses",
        "planned",
    )
    table.add_row(
        "sleeping_agents", "Many sessions, few active, tool-call pauses", "planned"
    )
    table.add_row(
        "recompute_crossover",
        "Find where recompute beats offloading per tier",
        "planned",
    )

    Console().print(table)


# ---------------------------------------------------------------------------
# Internal commands (used by generated manifests)
# ---------------------------------------------------------------------------


@main.command(name="_probe-node", hidden=True)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output JSON path for this node's probe results",
)
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def probe_node_cmd(output: str, quiet: bool) -> None:
    """[Internal] Run local tier measurements and write JSON result."""
    from memos.calibrate.probe import probe_node, probe_to_json

    result = probe_node(verbose=not quiet)
    json_str = probe_to_json(result)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_str)

    if not quiet:
        click.echo(f"Probe results written to {out_path}")


@main.command(name="_assemble", hidden=True)
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--platform", required=True, help="Platform name for metadata")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for the assembled hardware YAML",
)
def assemble_cmd(results_dir: str, platform: str, output: str) -> None:
    """[Internal] Assemble per-node probes + NCCL logs into final hardware YAML."""
    from memos.calibrate.assemble import assemble
    from memos.calibrate.platforms import load_platform

    plat = load_platform(platform)
    assemble(results_dir, plat, output=output)
    click.echo(f"Assembled hardware YAML written to {output}")
