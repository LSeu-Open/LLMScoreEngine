"""Performance instrumentation helpers for the shell runtime."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Optional

from .logging import get_logger


@dataclass(slots=True)
class PerformanceBudget:
    """Budget definition for a monitored section."""

    label: str
    threshold_ms: float
    description: str = ""


@dataclass(slots=True)
class PerformanceSample:
    """Single measurement sample recorded by the monitor."""

    label: str
    duration_ms: float


class PerformanceMonitor:
    """Tracks performance metrics for render/command loops."""

    def __init__(
        self,
        *,
        budgets: Optional[Dict[str, PerformanceBudget]] = None,
        logger=None,
    ) -> None:
        self._budgets = budgets or {}
        self._samples: Dict[str, List[PerformanceSample]] = defaultdict(list)
        self._violations: List[PerformanceSample] = []
        self._logger = logger or get_logger("llmscore.perf")

    @contextmanager
    def measure(self, label: str):
        """Context manager that records elapsed time for a section."""

        start = perf_counter()
        try:
            yield
        finally:
            duration_ms = (perf_counter() - start) * 1000
            self.record(label, duration_ms)

    def record(self, label: str, duration_ms: float) -> None:
        sample = PerformanceSample(label=label, duration_ms=duration_ms)
        self._samples[label].append(sample)
        budget = self._budgets.get(label)
        if budget and duration_ms > budget.threshold_ms:
            self._violations.append(sample)
            self._logger.warning(
                "Performance budget exceeded for %s (%.2fms > %.2fms)",
                label,
                duration_ms,
                budget.threshold_ms,
            )

    @property
    def violations(self) -> Iterable[PerformanceSample]:
        return tuple(self._violations)

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Return the latest samples along with their budgets."""

        report: Dict[str, Dict[str, Optional[float]]] = {}
        for label, samples in self._samples.items():
            if not samples:
                continue
            latest = samples[-1]
            budget = self._budgets.get(label)
            report[label] = {
                "last_ms": latest.duration_ms,
                "budget_ms": budget.threshold_ms if budget else None,
                "violations": sum(
                    1 for sample in samples if self._is_violation(label, sample)
                ),
            }
        return report

    def _is_violation(self, label: str, sample: PerformanceSample) -> bool:
        budget = self._budgets.get(label)
        return bool(budget and sample.duration_ms > budget.threshold_ms)


__all__ = ["PerformanceBudget", "PerformanceMonitor", "PerformanceSample"]
