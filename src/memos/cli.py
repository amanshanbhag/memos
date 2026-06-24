from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import click

from memos.hardware import load_hardware


@click.group()
def main() -> None:
    """memos - memory roofline and benchmarks for AI inference."""
    pass


@main.command()
@click.argument("workload_name")
@click.option(
    "--hw", required=True, type=click.Path(exists=True), help="Hardware config YAML"
)
@click.option("--model", required=True, help="Model name or path")
@click.option(
    "--output", default="results/", type=click.Path(), help="Output directory"
)
@click.option(
    "--tp", default=None, type=int, help="Tensor parallel size (default: hw.gpu_count)"
)
@click.option("--pp", default=None, type=int, help="Pipeline parallel size")
@click.option("--dp", default=None, type=int, help="Data parallel size")
@click.option("--cache-mode", default="cold", type=click.Choice(["cold", "warm"]))
@click.option("--context-lengths", default=None, help="Comma-separated context lengths")
@click.option("--repeats", default=3, type=int, help="Repeats per context length")
@click.option(
    "--output-tokens", default=128, type=int, help="Tokens to generate per request"
)
@click.option(
    "--engine-arg",
    multiple=True,
    help="Extra engine args as key=value (e.g. --engine-arg max_model_len=4096)",
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
    context_lengths: str | None,
    repeats: int,
    output_tokens: int,
    engine_arg: tuple[str, ...],
) -> None:
    """Run a benchmark workload."""
    from memos.environment import detect_environment
    from memos.metrics.throughput import ThroughputCollector
    from memos.results import save_result
    from memos.runners.vllm_runner import VLLMRunner
    from memos.workloads.context_sweep import ContextSweep

    hw_config = load_hardware(hw)

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

    # Parse context lengths
    ctx_lens = None
    if context_lengths:
        ctx_lens = [int(x.strip()) for x in context_lengths.split(",")]

    # Detect environment
    env = detect_environment()
    click.echo(
        f"GPU: {env.gpu_name} | Driver: {env.driver_version} | CUDA: {env.cuda_version}"
    )
    click.echo(f"Packages: {env.packages}")
    click.echo()

    # Setup runner
    runner = VLLMRunner()
    runner_kwargs = {}
    if tp is not None:
        runner_kwargs["tensor_parallel_size"] = tp
    if pp is not None:
        runner_kwargs["pipeline_parallel_size"] = pp
    if dp is not None:
        runner_kwargs["data_parallel_size"] = dp
    runner_kwargs["enable_prefix_caching"] = cache_mode == "warm"

    for arg in engine_arg:
        k, _, v = arg.partition("=")
        # Try to parse as int/float/bool, fall back to string
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

    # Setup collectors
    collectors = [ThroughputCollector()]

    # Create and run workload
    workload = workloads[workload_name](
        context_lengths=ctx_lens,
        output_tokens=output_tokens,
        repeats=repeats,
        cache_mode=cache_mode,
    )
    click.echo(f"Running: {workload.description()}")
    click.echo()

    import dataclasses

    result = workload.run(runner, hw_config, collectors, model=model)
    result.environment = dataclasses.asdict(env)

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{workload_name}_{hw_config.name}_{timestamp}.json"
    out_path = save_result(result, Path(output) / filename)
    click.echo(f"\nResults saved to {out_path}")

    # Print summary
    tps_samples = [m for m in result.metrics if m.name == "tokens_per_sec"]
    if tps_samples:
        by_ctx = {}
        for s in tps_samples:
            ctx = s.context.get("context_length", 0)
            by_ctx.setdefault(ctx, []).append(s.value)
        click.echo("\nTokens/sec by context length:")
        for ctx in sorted(by_ctx):
            vals = by_ctx[ctx]
            avg = sum(vals) / len(vals)
            click.echo(f"  {ctx:>8} tokens: {avg:>10.1f} tok/s")

    runner.shutdown()


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option(
    "--hw", required=True, type=click.Path(exists=True), help="Hardware config YAML"
)
@click.option("--output", default=None, type=click.Path(), help="Save plot to file")
def roofline(results_dir: str, hw: str, output: str | None) -> None:
    """Generate a memory roofline plot from benchmark results."""
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

        by_ctx = {}
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

            ceilings = compute_ceilings(
                hw=hw_config,
                flops_per_token=float(hw_config.metadata.get("peak_flops", 1e15))
                / 1000,
                bytes_per_token=ctx * 2,  # rough: 2 bytes per token of KV cache (fp16)
            )
            all_ceilings.append(ceilings)
            all_labels.append(f"{ctx // 1024}K")
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
