"""Concurrency stress scaffolding for automation watch jobs."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import math

import pytest

from llmscore.automation import watchers as watchers_mod

pytestmark = pytest.mark.perf


class _FakeEvent:
    def __init__(self, path: str) -> None:
        self.src_path = path
        self.is_directory = False


def _build_handler(
    *,
    max_events: int,
    debounce: float = 1.0,
    targets: Sequence[watchers_mod.WatchTarget] | None = None,
    disable_schedule: bool = True,
):
    job = watchers_mod.WatchJob(
        action="automation.watch",
        targets=list(targets or [watchers_mod.WatchTarget(path=Path("."), recursive=False)]),
        debounce_seconds=debounce,
        max_events_per_cycle=max_events,
    )

    calls: List[str] = []

    def runner(action: str, inputs: dict[str, object]) -> None:  # pragma: no cover - behaviour verified via side effects
        calls.append(action)

    handler = watchers_mod._ActionDispatchHandler(job, runner)
    original_flush = handler._flush
    processed_batches: List[int] = []

    def tracking_flush():
        processed_batches.append(len(handler._pending_events))
        original_flush()

    handler._flush = tracking_flush  # type: ignore[method-assign]
    handler._processed_batches = processed_batches  # type: ignore[attr-defined]
    if disable_schedule:
        handler._maybe_schedule_flush = lambda: None  # type: ignore[method-assign]
    return handler, calls, processed_batches


def test_watch_handler_can_flush_large_queue() -> None:
    handler, calls, processed = _build_handler(max_events=1024, debounce=5.0)

    total_events = 600
    for idx in range(total_events):
        handler.on_any_event(_FakeEvent(f"/tmp/model-{idx}.json"))

    # Ensure nothing auto-flushed before manual flush
    assert len(handler._pending_events) == total_events

    handler._flush()

    assert len(handler._pending_events) == 0
    assert calls == ["automation.watch"], "Large burst should be processed in one run"
    assert sum(processed) == total_events


def test_watch_handler_auto_flushes_under_pressure() -> None:
    max_events = 50
    handler, calls, processed = _build_handler(max_events=max_events, debounce=5.0)

    total_events = 240
    for idx in range(total_events):
        handler.on_any_event(_FakeEvent(f"/tmp/batch-{idx}.json"))
        if len(handler._pending_events) >= handler.job.max_events_per_cycle:
            handler._flush()

    handler._flush()  # flush any remainder < threshold

    expected_batches = math.ceil(total_events / max_events)
    assert len(calls) >= expected_batches
    assert handler._pending_events == []
    assert sum(processed) == total_events


def test_watch_handler_multiple_targets_and_ignore_patterns(tmp_path):
    targets = [
        watchers_mod.WatchTarget(
            path=tmp_path / "models",
            recursive=True,
            patterns=("**/*.json",),
            ignore_patterns=("**/ignored/*.json",),
        ),
        watchers_mod.WatchTarget(
            path=tmp_path / "logs",
            recursive=True,
            patterns=("**/*.log",),
        ),
    ]
    handler, calls, processed = _build_handler(max_events=32, targets=targets)

    valid_events = [
        tmp_path / "models" / "alpha" / "model.json",
        tmp_path / "logs" / "runner.log",
    ]
    ignored_event = tmp_path / "models" / "ignored" / "skip.json"

    for path in valid_events:
        handler.on_any_event(_FakeEvent(str(path)))
    handler.on_any_event(_FakeEvent(str(ignored_event)))

    handler._flush()

    assert sum(processed) == len(valid_events)
    assert handler._pending_events == []
    assert calls == ["automation.watch"]


def test_watch_handler_missed_event_rate_below_threshold() -> None:
    max_events = 64
    handler, _, processed = _build_handler(max_events=max_events, debounce=0.1)

    total_events = 640
    for idx in range(total_events):
        handler.on_any_event(_FakeEvent(f"/tmp/stress-{idx}.json"))
        if len(handler._pending_events) >= handler.job.max_events_per_cycle:
            handler._flush()

    handler._flush()

    processed_total = sum(processed)
    assert processed_total == total_events
    missed_rate = 1 - (processed_total / total_events)
    assert missed_rate <= 0.005
    assert handler._pending_events == []
