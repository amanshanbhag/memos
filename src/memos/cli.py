from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from memos.hardware import load_hardware
from memos.types import InferenceConfig


def _parse_engine_arg_value(raw: str) -> int | float | bool | str:
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _infer_kv_dtype_bytes(engine_args: dict[str, object]) -> float:
    kv_dtype = str(engine_args.get("kv_cache_dtype", "fp16")).lower()
    if "fp8" in kv_dtype or kv_dtype in {"int8", "uint8"}:
        return 1.0
    if "int4" in kv_dtype or "fp4" in kv_dtype:
        return 0.5
    return 2.0


def _infer_weight_dtype_bytes(engine_args: dict[str, object]) -> float:
    quant = str(engine_args.get("quantization", "")).lower()
    dtype = str(engine_args.get("dtype", "fp16")).lower()
    if any(k in quant for k in ("awq", "gptq", "int4", "fp4")):
        return 0.5
    if any(k in quant for k in ("int8", "fp8")):
        return 1.0
    if "fp8" in dtype:
        return 1.0
    return 2.0


def _infer_precision(inference: InferenceConfig | None) -> str:
    if inference is None:
        return "fp16"
    act = inference.activation_dtype.lower()
    if "fp4" in act:
        return "fp4"
    if "fp8" in act:
        return "fp8"
    if "int8" in act:
        return "int8"
    return "fp16"


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
    "--batch",
    default=1,
    type=int,
    help="Batch size (concurrent requests per generate call)",
)
@click.option(
    "--batch-mode",
    default="static",
    type=click.Choice(["static", "concurrency", "saturated"]),
    help=(
        "static: B requests per repeat; concurrency: sustained waves of B; "
        "saturated: deep queue with running batch pinned at B (max_num_seqs=B)"
    ),
)
@click.option(
    "--max-num-seqs",
    default=None,
    type=int,
    help="Pin vLLM max_num_seqs (running batch cap). Auto-set to --batch in "
    "saturated mode if unset.",
)
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
    batch: int,
    batch_mode: str,
    max_num_seqs: int | None,
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

    # Saturated mode pins the running batch by capping max_num_seqs at B.
    effective_max_num_seqs = max_num_seqs
    if effective_max_num_seqs is None and batch_mode == "saturated":
        effective_max_num_seqs = batch

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
            batch=batch,
            batch_mode=batch_mode,
            max_num_seqs=effective_max_num_seqs,
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
    if effective_max_num_seqs is not None:
        runner_kwargs["max_num_seqs"] = effective_max_num_seqs
    runner_kwargs["enable_prefix_caching"] = cache_mode == "warm"

    parsed_engine_args: dict[str, object] = {}
    for arg in engine_arg:
        k, _, v = arg.partition("=")
        if not k:
            continue
        parsed_val = _parse_engine_arg_value(v)
        parsed_engine_args[k] = parsed_val
        runner_kwargs[k] = parsed_val

    runner.setup(model, hw_config, **runner_kwargs)

    collectors = [ThroughputCollector()]

    osls = [int(x.strip()) for x in output_tokens.split(",")]
    workload = workloads[workload_name](
        isls=isls,
        osls=osls,
        repeats=repeats,
        cache_mode=cache_mode,
        batch=batch,
        batch_mode=batch_mode,
    )
    click.echo(f"Running: {workload.description()}")
    click.echo()

    import dataclasses

    result = workload.run(runner, hw_config, collectors, model=model)
    result.environment = dataclasses.asdict(env)
    result.raw["model_profile"] = profile.to_dict()
    result.inference_config = InferenceConfig(
        tp=tp if tp is not None else hw_config.gpu_count,
        pp=pp if pp is not None else 1,
        dp=dp if dp is not None else 1,
        batch_size=batch,
        max_num_seqs=effective_max_num_seqs or 0,
        cache_mode=cache_mode,
        weight_dtype_bytes=_infer_weight_dtype_bytes(parsed_engine_args),
        kv_dtype_bytes=_infer_kv_dtype_bytes(parsed_engine_args),
        weight_group_size=int(parsed_engine_args.get("weight_group_size", 0) or 0),
        activation_dtype=str(parsed_engine_args.get("dtype", "fp16")),
        speculative=(
            "speculative_model" in parsed_engine_args
            or int(parsed_engine_args.get("num_speculative_tokens", 0) or 0) > 0
        ),
        mtp=(
            "mtp" in parsed_engine_args
            or "medusa" in parsed_engine_args
            or int(parsed_engine_args.get("num_lookahead_slots", 0) or 0) > 0
        ),
        engine_args=parsed_engine_args,
    )

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
# memos bench-serve
# ---------------------------------------------------------------------------


@main.command(name="bench-serve")
@click.option("--hw", required=True, type=click.Path(), help="Hardware config YAML")
@click.option("--model", required=True, help="Model name or path")
@click.option(
    "--output", "-o", default="results/", type=click.Path(), help="Output directory"
)
@click.option(
    "--tp", default=None, type=int, help="Tensor parallel size (default: hw.gpu_count)"
)
@click.option("--isl", default=1024, type=int, help="Random dataset input length")
@click.option("--osl", default=128, type=int, help="Random dataset output length")
@click.option("--num-prompts", default=500, type=int, help="Requests per grid point")
@click.option(
    "--warmup-prompts",
    default=64,
    type=int,
    help="Discarded warmup requests before the grid (0 disables) to absorb "
    "one-time compile/graph-capture stalls",
)
@click.option(
    "--request-rate",
    default="inf",
    help="Offered QPS per point, comma-separated ('inf' = unthrottled, e.g. 4,8,16,inf)",
)
@click.option(
    "--max-concurrency",
    default=None,
    help="Optional in-flight request cap(s), comma-separated (client-side)",
)
@click.option("--dataset", default="random", help="vllm bench dataset name")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option("--percentiles", default="90,95,99", help="Latency percentiles to report")
@click.option(
    "--engine-arg",
    multiple=True,
    help="Extra engine args as key=value (e.g. --engine-arg kv_cache_dtype=fp8)",
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
@click.option("--container-image", default=None, help="Container image")
@click.option("--container-mounts", default=None, help="Container bind mounts (SLURM)")
@click.option("--nodelist", default=None, help="Comma-separated node names")
@click.option("--namespace", default=None, help="K8s namespace")
@click.option("--pvc", default=None, help="K8s PVC for shared results storage")
@click.option("--env", multiple=True, help="Environment variables for container")
@click.option("--slurm-arg", multiple=True, help="Extra SLURM directives")
def bench_serve(
    hw: str,
    model: str,
    output: str,
    tp: int | None,
    isl: int,
    osl: int,
    num_prompts: int,
    warmup_prompts: int,
    request_rate: str,
    max_concurrency: str | None,
    dataset: str,
    port: int,
    percentiles: str,
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
    """Open-loop serving benchmark: TTFT/TPOT/e2e SLO curves under an arrival rate.

    Wraps `vllm serve` + `vllm bench serve`. Sweeps --request-rate (and optional
    --max-concurrency) at a fixed ISL/OSL, capturing latency percentiles,
    throughput, and in-flight pressure (running/waiting queue, KV usage,
    preemptions) -- the regime steady-state batch sweeps cannot reach.
    """
    hw_config = load_hardware(hw)
    rates = [r.strip() for r in request_rate.split(",") if r.strip()]
    concs = (
        [int(c.strip()) for c in max_concurrency.split(",") if c.strip()]
        if max_concurrency
        else None
    )

    # --- Submission mode: generate manifest and exit ---
    if scheduler:
        from memos.calibrate.manifest import render_bench_serve_manifest

        manifest_str = render_bench_serve_manifest(
            scheduler=scheduler,
            hw_path=hw,
            model=model,
            gpus_per_node=hw_config.gpu_count,
            output_dir=output,
            nodes=nodes,
            tp=tp,
            isl=isl,
            osl=osl,
            num_prompts=num_prompts,
            warmup_prompts=warmup_prompts,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            dataset=dataset,
            port=port,
            percentiles=percentiles,
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
            hint = "sbatch" if scheduler == "slurm" else "kubectl apply -f"
            click.echo(f"Submit with: {hint} {out_path}")
        else:
            click.echo(manifest_str)
        return

    # --- Local execution mode ---
    import dataclasses

    from memos.environment import detect_environment
    from memos.results import save_result
    from memos.serving.bench import run_bench_serve

    parsed_engine_args: dict[str, object] = {}
    for arg in engine_arg:
        k, _, v = arg.partition("=")
        if k:
            parsed_engine_args[k] = _parse_engine_arg_value(v)

    effective_tp = tp if tp is not None else hw_config.gpu_count
    env_map = dict(e.split("=", 1) for e in env if "=" in e)

    click.echo(f"Serving benchmark: {model} on {hw_config.name}")
    click.echo(f"  tp={effective_tp} isl={isl} osl={osl} num_prompts={num_prompts}")
    click.echo(f"  request_rate={rates} max_concurrency={concs}")
    click.echo()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    server_log = str(Path(output) / f"vllm_serve_{hw_config.name}_{timestamp}.log")

    result = run_bench_serve(
        model=model,
        hw=hw_config,
        tp=effective_tp,
        isl=isl,
        osl=osl,
        num_prompts=num_prompts,
        request_rates=rates,
        max_concurrency=concs,
        engine_args=parsed_engine_args,
        dataset=dataset,
        port=port,
        percentiles=percentiles,
        server_log=server_log,
        env=env_map or None,
        warmup_prompts=warmup_prompts,
    )

    result.environment = dataclasses.asdict(detect_environment())
    result.inference_config = InferenceConfig(
        tp=effective_tp,
        weight_dtype_bytes=_infer_weight_dtype_bytes(parsed_engine_args),
        kv_dtype_bytes=_infer_kv_dtype_bytes(parsed_engine_args),
        activation_dtype=str(parsed_engine_args.get("dtype", "fp16")),
        engine_args=parsed_engine_args,
    )

    filename = f"bench_serve_{hw_config.name}_{timestamp}.json"
    out_path = save_result(result, Path(output) / filename)
    click.echo(f"\nResults saved to {out_path}")

    _echo_serving_summary(result)


@main.command(name="serve-plot")
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path(), help="Save plot to file")
@click.option(
    "--ttft-slo", default=500.0, type=float, help="p99 TTFT SLO (ms) for the knee"
)
@click.option(
    "--tpot-slo", default=50.0, type=float, help="p99 TPOT SLO (ms) for the knee"
)
def serve_plot(
    results_path: str, output: str | None, ttft_slo: float, tpot_slo: float
) -> None:
    """Plot serving SLO curves + print the max-throughput-under-SLO knee per config.

    Reads bench_serve_*.json under RESULTS_PATH (one series per config dir) and
    renders latency/pressure vs achieved-throughput curves.
    """
    from memos.serving.plot import find_knee, load_serving_series, plot_serving

    series = load_serving_series(results_path)
    if not series:
        click.echo(f"No bench_serve_*.json found under {results_path}")
        return

    click.echo(f"Loaded {len(series)} serving config(s)")
    plot_serving(
        series,
        title=f"Serving SLO curves ({Path(results_path).name})",
        output=output,
        ttft_slo_ms=ttft_slo,
        tpot_slo_ms=tpot_slo,
    )
    if output:
        click.echo(f"Plot saved to {output}")

    click.echo(
        f"\nSLO knee (max achieved req/s with p99 TTFT<{ttft_slo:g}ms "
        f"and p99 TPOT<{tpot_slo:g}ms):"
    )
    for s in sorted(series, key=lambda x: x.name):
        knee = find_knee(s, ttft_slo, tpot_slo)
        if knee:
            click.echo(
                f"  {s.name:<40} {knee.request_throughput:6.1f} req/s "
                f"({knee.output_throughput:8.0f} tok/s) @ rate={knee.request_rate}"
            )
        else:
            click.echo(f"  {s.name:<40} (no point meets SLO)")


def _echo_serving_summary(result) -> None:
    """Print a compact SLO/throughput table grouped by request rate."""
    by_rate: dict[str, dict[str, float]] = {}
    for m in result.metrics:
        rate = str(m.context.get("request_rate", "?"))
        by_rate.setdefault(rate, {})[m.name] = m.value
    if not by_rate:
        return
    click.echo("\nOffered QPS -> throughput / p99 TTFT / p99 TPOT / preemptions:")
    for rate, vals in by_rate.items():
        click.echo(
            f"  rate={rate:>4}: "
            f"{vals.get('output_throughput', 0):>8.0f} tok/s  "
            f"ttft_p99={vals.get('p99_ttft_ms', 0):>8.1f} ms  "
            f"tpot_p99={vals.get('p99_tpot_ms', 0):>7.1f} ms  "
            f"preempt={vals.get('num_preemptions', 0):>4.0f}"
        )


# ---------------------------------------------------------------------------
# memos roofline
# ---------------------------------------------------------------------------


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option(
    "--hw", required=True, type=click.Path(exists=True), help="Hardware config YAML"
)
@click.option("--output", default=None, type=click.Path(), help="Save plot to file")
def roofline(results_path: str, hw: str, output: str | None) -> None:
    """Generate a classic roofline plot from benchmark results."""
    import dataclasses

    from memos.model_profile import ModelProfile
    from memos.results import load_result
    from memos.roofline.model import compute_ceilings
    from memos.roofline.plot import RooflinePoint, plot_roofline

    hw_config = load_hardware(hw)
    input_path = Path(results_path)
    if input_path.is_file():
        result_files = [input_path]
    else:
        result_files = sorted(input_path.rglob("*.json"))

    if not result_files:
        click.echo(f"No result JSON files found in {results_path}")
        return

    click.echo(f"Found {len(result_files)} result file(s)")
    click.echo(f"Hardware: {hw_config.name}")

    points: list[RooflinePoint] = []
    all_precisions: set[str] = set()
    all_ceiling_checks: list[tuple[str, float, float, str]] = []

    for rf in result_files:
        result = load_result(rf)
        profile_dict = result.raw.get("model_profile")
        if not profile_dict:
            continue
        profile = ModelProfile.from_dict(profile_dict)

        inference = result.inference_config or InferenceConfig()
        precision = _infer_precision(inference)
        all_precisions.add(precision)
        config_name = rf.parent.name

        tps_samples = [m for m in result.metrics if m.name == "tokens_per_sec"]
        by_combo: dict[tuple[int, int], list[float]] = {}
        for sample in tps_samples:
            isl = int(sample.context.get("context_length", 0))
            osl = int(sample.context.get("output_tokens", 0))
            by_combo.setdefault((isl, osl), []).append(sample.value)

        # Measured average running batch (sampled in-flight) -> the point's true
        # position on the AI axis. Falls back to the nominal submitted batch.
        rb_by_combo: dict[tuple[int, int], list[float]] = {}
        for sample in (m for m in result.metrics if m.name == "running_batch_avg"):
            isl = int(sample.context.get("context_length", 0))
            osl = int(sample.context.get("output_tokens", 0))
            rb_by_combo.setdefault((isl, osl), []).append(sample.value)

        for (isl, osl), vals in sorted(by_combo.items()):
            if not vals or isl <= 0:
                continue
            avg_tps = sum(vals) / len(vals)

            rb_vals = [v for v in rb_by_combo.get((isl, osl), []) if v > 0]
            if rb_vals:
                measured_batch = max(1, round(sum(rb_vals) / len(rb_vals)))
                eff_inference = dataclasses.replace(
                    inference, batch_size=measured_batch
                )
            else:
                eff_inference = inference

            flops_per_token = profile.flops_per_token(
                seq=isl, inference_config=eff_inference
            )
            bytes_per_token = profile.bytes_per_token(
                seq=isl, inference_config=eff_inference
            )
            if bytes_per_token <= 0 or flops_per_token <= 0:
                continue

            weight_bytes = profile.weight_bytes()
            weight_scaled = weight_bytes * (
                eff_inference.weight_dtype_bytes / max(float(profile.dtype_bytes), 1e-9)
            )
            if eff_inference.weight_group_size > 0:
                weight_scaled *= 1.0 + (
                    4.0
                    / (
                        eff_inference.weight_group_size
                        * max(float(eff_inference.weight_dtype_bytes), 1e-9)
                    )
                )
            tp = max(eff_inference.tp, 1)
            batch = max(eff_inference.batch_size, 1)
            working_set = (weight_scaled / tp) + (
                profile.kv_cache_bytes(seq=isl, batch=batch) / tp
            )

            ceilings = compute_ceilings(
                hw=hw_config,
                flops_per_token=flops_per_token,
                bytes_per_token=bytes_per_token,
                working_set_bytes=working_set,
                precision=precision,
            )
            ai = flops_per_token / bytes_per_token
            measured_flops = avg_tps * flops_per_token
            points.append(
                RooflinePoint(
                    label=f"{config_name}:{isl}x{osl}",
                    arithmetic_intensity=ai,
                    measured_flops_per_sec=measured_flops,
                    series=config_name,
                )
            )
            all_ceiling_checks.append(
                (
                    f"{config_name}:{isl}x{osl}",
                    avg_tps,
                    min(ceilings.compute_ceiling, ceilings.bandwidth_ceiling),
                    ceilings.bottleneck,
                )
            )

    if not points:
        click.echo("No roofline-eligible data points found.")
        return

    tier_bandwidths = {t.name: t.bandwidth_gbps for t in hw_config.tiers}
    primary_precision = next(iter(sorted(all_precisions))) if all_precisions else "fp16"
    peak_flops = float(
        hw_config.metadata.get(
            f"peak_flops_{primary_precision}",
            hw_config.metadata.get("peak_flops", 0.0),
        )
    )

    plot_roofline(
        points=points,
        tier_bandwidths_gbps=tier_bandwidths,
        peak_flops=peak_flops,
        title=f"Roofline ({primary_precision.upper()}) - {hw_config.name}",
        output=output,
    )
    if output:
        click.echo(f"Plot saved to {output}")
    else:
        click.echo("Plot displayed.")

    violations = [
        (name, measured, ceiling, bottleneck)
        for name, measured, ceiling, bottleneck in all_ceiling_checks
        if measured > ceiling
    ]
    if violations:
        click.echo(
            f"[warn] {len(violations)} points exceed modeled token/s ceilings; "
            "model likely still undercounting bytes for those configs."
        )


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
