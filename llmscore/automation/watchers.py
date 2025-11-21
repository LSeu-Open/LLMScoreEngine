"""Filesystem watch utilities for triggering actions on change."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..utils.logging import get_logger

try:  # pragma: no cover - optional dependency guard
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "watchdog is required for watch mode. Install with 'pip install "
        "watchdog'."
    ) from exc

LOGGER = get_logger("llmscore.automation.watchers")

ActionRunner = Callable[[str, Dict[str, Any]], None]
NotificationCallback = Callable[[str], None]


@dataclass(slots=True)
class WatchTarget:
    """Configuration for a directory watch target."""

    path: Path
    recursive: bool = True
    patterns: Sequence[str] = ("*",)
    ignore_patterns: Sequence[str] = ()

    def normalized_path(self) -> Path:
        return self.path.expanduser().resolve()


@dataclass(slots=True)
class WatchJob:
    """Defines the action to trigger when filesystem events occur."""

    action: str
    targets: Sequence[WatchTarget]
    inputs: Dict[str, Any] = field(default_factory=dict)
    debounce_seconds: float = 1.0
    max_events_per_cycle: int = 5


class _ActionDispatchHandler(FileSystemEventHandler):
    """Internal watchdog handler that debounces and triggers actions."""

    def __init__(
        self,
        job: WatchJob,
        runner: ActionRunner,
        notify: Optional[NotificationCallback] = None,
    ) -> None:
        self.job = job
        self.runner = runner
        self.notify = notify
        # ``on_any_event`` may call ``_flush`` while still holding this lock
        # (when a burst of events reaches ``max_events_per_cycle``). A regular
        # ``Lock`` would deadlock in that situation, so use ``RLock`` to allow
        # the same thread to re-enter.
        self._lock = threading.RLock()
        self._last_trigger: float = 0.0
        self._pending_events: List[str] = []

    def on_any_event(self, event: FileSystemEvent) -> None:  # pragma: no cover
        if event.is_directory:
            return
        with self._lock:
            if not self._should_record(event.src_path):
                return
            self._pending_events.append(event.src_path)
            if len(self._pending_events) >= self.job.max_events_per_cycle:
                self._flush()
            else:
                self._maybe_schedule_flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _should_record(self, src_path: str) -> bool:
        if not self.job.targets:
            return False
        path_obj = Path(src_path)
        for target in self.job.targets:
            if self._matches_target(target, path_obj):
                return True
        return False

    @staticmethod
    def _matches_target(target: WatchTarget, path_obj: Path) -> bool:
        for pattern in target.ignore_patterns:
            if path_obj.match(pattern):
                return False
        if not target.patterns:
            return True
        return any(path_obj.match(pattern) for pattern in target.patterns)

    def _maybe_schedule_flush(self) -> None:
        now = time.time()
        elapsed = now - self._last_trigger
        if elapsed >= self.job.debounce_seconds:
            self._flush()
            return
        delay = self.job.debounce_seconds - elapsed
        timer = threading.Timer(delay, self._flush)
        timer.daemon = True
        timer.start()

    def _flush(self) -> None:
        with self._lock:
            if not self._pending_events:
                return
            events = list(self._pending_events)
            self._pending_events.clear()
            self._last_trigger = time.time()
        LOGGER.info(
            "Triggering action %s due to %d filesystem events",
            self.job.action,
            len(events),
        )
        if self.notify:
            message = (
                f"Detected {len(events)} changes. Running {self.job.action}..."
            )
            self.notify(message)
        try:
            self.runner(self.job.action, dict(self.job.inputs))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Watch job failed: %s", exc)
            if self.notify:
                self.notify(f"Action {self.job.action} failed: {exc}")
        else:
            if self.notify:
                self.notify(f"Action {self.job.action} completed.")


class WatchService:
    """Controls lifecycle of filesystem watchers for automation."""

    def __init__(
        self,
        job: WatchJob,
        runner: ActionRunner,
        *,
        notify: Optional[NotificationCallback] = None,
    ) -> None:
        self.job = job
        self.runner = runner
        self.notify = notify
        self._observer: Optional[Observer] = None

    def start(self) -> None:
        if self._observer is not None:
            LOGGER.debug("WatchService already running")
            return
        observer = Observer()
        handler = _ActionDispatchHandler(
            self.job,
            self.runner,
            notify=self.notify,
        )
        for target in self.job.targets:
            normalized = target.normalized_path()
            observer.schedule(
                handler,
                str(normalized),
                recursive=target.recursive,
            )
            LOGGER.info(
                "Watching %s (recursive=%s)",
                normalized,
                target.recursive,
            )
        observer.start()
        self._observer = observer

    def stop(self) -> None:
        observer = self._observer
        if observer is None:
            return
        observer.stop()
        observer.join(timeout=5)
        self._observer = None
        LOGGER.info("WatchService stopped")

    def run_forever(self) -> None:  # pragma: no cover - blocking loop
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            LOGGER.info("WatchService interrupted by user")
        finally:
            self.stop()


def build_watch_job(
    paths: Iterable[str | Path],
    *,
    action: str,
    inputs: Optional[Dict[str, Any]] = None,
    recursive: bool = True,
    patterns: Optional[Sequence[str]] = None,
    ignore_patterns: Optional[Sequence[str]] = None,
    debounce_seconds: float = 1.0,
) -> WatchJob:
    """Helper to build a :class:`WatchJob` from simple parameters."""

    targets = [
        WatchTarget(
            path=Path(path),
            recursive=recursive,
            patterns=tuple(patterns or ("*",)),
            ignore_patterns=tuple(ignore_patterns or ()),
        )
        for path in paths
    ]
    return WatchJob(
        action=action,
        targets=targets,
        inputs=dict(inputs or {}),
        debounce_seconds=debounce_seconds,
    )


__all__ = ["WatchJob", "WatchService", "WatchTarget", "build_watch_job"]
