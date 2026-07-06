"""vLLM OpenAI server lifecycle + in-flight Prometheus sampling.

`VLLMServer` launches `vllm serve` as a subprocess in its own process group,
waits for `/health`, and tears the whole group down on exit. `PrometheusPoller`
samples the server's `/metrics` endpoint on a background thread during a bench
point so we capture the pressure signals (running/waiting queue, KV usage,
preemptions) that reveal when the arrival rate has pushed the scheduler past
admission control.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any


def _emit_engine_args(engine_args: dict[str, Any]) -> list[str]:
    """Translate engine args into `vllm serve` CLI flags.

    Booleans become store_true flags (True -> `--flag`, False -> omitted);
    everything else becomes `--flag value`. Underscores map to dashes.
    """
    argv: list[str] = []
    for key, value in engine_args.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


class VLLMServer:
    """Context manager around a `vllm serve` subprocess."""

    def __init__(
        self,
        model: str,
        tp: int = 1,
        port: int = 8000,
        host: str = "127.0.0.1",
        engine_args: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
        startup_timeout_s: float = 1800.0,
        log_path: str | None = None,
    ) -> None:
        self._model = model
        self._tp = max(int(tp), 1)
        self._port = int(port)
        self._host = host
        self._engine_args = engine_args or {}
        self._env = env
        self._startup_timeout_s = startup_timeout_s
        self._log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def metrics_url(self) -> str:
        return f"{self.base_url}/metrics"

    def _build_command(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self._model,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--tensor-parallel-size",
            str(self._tp),
            # V1 exposes num_requests_running/waiting + cache usage on /metrics.
            "--disable-log-requests",
        ]
        cmd.extend(_emit_engine_args(self._engine_args))
        return cmd

    def start(self) -> "VLLMServer":
        env = dict(os.environ)
        if self._env:
            env.update(self._env)

        if self._log_path:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            self._log_fh = open(self._log_path, "w")
            stdout: Any = self._log_fh
            stderr: Any = subprocess.STDOUT
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        # start_new_session so we can kill the whole vLLM process tree (the V1
        # engine spawns worker processes) via the process group on teardown.
        self._proc = subprocess.Popen(
            self._build_command(),
            stdout=stdout,
            stderr=stderr,
            env=env,
            start_new_session=True,
        )
        self._wait_healthy()
        return self

    def _wait_healthy(self) -> None:
        assert self._proc is not None
        deadline = time.time() + self._startup_timeout_s
        health = f"{self.base_url}/health"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited (code {self._proc.returncode}) during startup; "
                    f"see log {self._log_path or '(no log)'}"
                )
            try:
                with urllib.request.urlopen(health, timeout=2.0) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(2.0)
        self.stop()
        raise TimeoutError(
            f"vllm serve did not become healthy within {self._startup_timeout_s:.0f}s"
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGINT)
                try:
                    self._proc.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    self._proc.wait(timeout=10.0)
            except (ProcessLookupError, OSError):
                pass
        self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> "VLLMServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class PrometheusPoller:
    """Samples a vLLM server's /metrics endpoint on a background thread.

    Construct a fresh poller per bench point (as a context manager) so the peak
    running/waiting queue and KV/CPU cache usage reflect that point, and
    num_preemptions is reported as the delta observed across the point.
    """

    _RUNNING = ("vllm:num_requests_running",)
    _WAITING = ("vllm:num_requests_waiting",)
    _KV = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    _CPU = ("vllm:cpu_cache_usage_perc",)
    _PREEMPT = ("vllm:num_preemptions_total", "vllm:num_preemptions")

    def __init__(self, metrics_url: str, interval_s: float = 0.25) -> None:
        self._url = metrics_url
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running: list[float] = []
        self._waiting: list[float] = []
        self._kv: list[float] = []
        self._cpu: list[float] = []
        self._preempt_first: float | None = None
        self._preempt_last: float | None = None

    def _read(self) -> dict[str, float]:
        with urllib.request.urlopen(self._url, timeout=2.0) as resp:
            body = resp.read().decode("utf-8", "replace")
        return parse_prometheus_metrics(body)

    @staticmethod
    def _first(metrics: dict[str, float], keys: tuple[str, ...]) -> float | None:
        for k in keys:
            if k in metrics:
                return metrics[k]
        return None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                m = self._read()
                for keys, buf in (
                    (self._RUNNING, self._running),
                    (self._WAITING, self._waiting),
                    (self._KV, self._kv),
                    (self._CPU, self._cpu),
                ):
                    v = self._first(m, keys)
                    if v is not None:
                        buf.append(v)
                p = self._first(m, self._PREEMPT)
                if p is not None:
                    if self._preempt_first is None:
                        self._preempt_first = p
                    self._preempt_last = p
            except Exception:
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> "PrometheusPoller":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, float]:
        preempt = 0.0
        if self._preempt_first is not None and self._preempt_last is not None:
            preempt = max(self._preempt_last - self._preempt_first, 0.0)
        return {
            "running_batch_peak": max(self._running) if self._running else 0.0,
            "waiting_peak": max(self._waiting) if self._waiting else 0.0,
            "kv_cache_usage_peak": max(self._kv) if self._kv else 0.0,
            "cpu_cache_usage_peak": max(self._cpu) if self._cpu else 0.0,
            "num_preemptions": preempt,
        }


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text-format exposition into {metric_name: value}.

    Label sets are stripped (bare metric name kept); when a metric appears with
    multiple label sets the last-seen sample wins. HELP/TYPE/comment and blank
    lines are ignored. Defensive: malformed lines are skipped.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        brace = line.find("{")
        if brace != -1:
            name = line[:brace]
            close = line.rfind("}")
            rest = line[close + 1 :] if close != -1 else ""
        else:
            name, _, rest = line.partition(" ")
        rest = rest.strip()
        if not rest:
            continue
        token = rest.split()[0]
        try:
            out[name.strip()] = float(token)
        except ValueError:
            continue
    return out
