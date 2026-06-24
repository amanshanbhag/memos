from __future__ import annotations

from abc import ABC, abstractmethod

from memos.metrics.base import MetricCollector
from memos.runners.base import Runner
from memos.types import BenchmarkResult, HardwareConfig


class Workload(ABC):
    """A benchmark scenario that drives a Runner and collects metrics."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier for this workload."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this workload measures."""
        ...

    @abstractmethod
    def run(
        self,
        runner: Runner,
        hw: HardwareConfig,
        collectors: list[MetricCollector],
    ) -> BenchmarkResult:
        """Execute the workload and return results."""
        ...
