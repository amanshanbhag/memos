from __future__ import annotations

import json
from pathlib import Path

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
def run(workload_name: str, hw: str, model: str, output: str) -> None:
    """Run a benchmark workload."""
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
    click.echo()
    click.echo(
        "Runner not yet implemented. This will work once VLLMRunner is connected."
    )


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option(
    "--hw", required=True, type=click.Path(exists=True), help="Hardware config YAML"
)
@click.option("--output", default=None, type=click.Path(), help="Save plot to file")
def roofline(results_dir: str, hw: str, output: str | None) -> None:
    """Generate a memory roofline plot from benchmark results."""
    hw_config = load_hardware(hw)
    results_path = Path(results_dir)

    result_files = list(results_path.glob("*.json"))
    if not result_files:
        click.echo(f"No result JSON files found in {results_dir}")
        return

    click.echo(f"Found {len(result_files)} result file(s)")
    click.echo(f"Hardware: {hw_config.name}")
    click.echo()
    click.echo("Roofline plotting not yet connected to result data. Coming soon.")


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
        "stub",
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
