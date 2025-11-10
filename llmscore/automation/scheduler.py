"""Scheduling and notification helpers for automation workflows."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from ..utils.logging import get_logger

try:  # pragma: no cover - optional dependency guard
    from croniter import croniter
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "croniter is required for scheduling. Install with 'pip install "
        "croniter'."
    ) from exc

try:  # pragma: no cover - optional dependency guard
    import httpx
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "httpx is required for webhook notifications. Install with 'pip "
        "install httpx'."
    ) from exc

LOGGER = get_logger("llmscore.automation.scheduler")

ActionRunner = Callable[[str, Dict[str, Any]], None]


@dataclass(slots=True)
class ScheduleJob:
    """Defines a cron-driven action execution."""

    action: str
    cron: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    tz: timezone = timezone.utc

    def next_run(self, *, from_time: Optional[datetime] = None) -> datetime:
        reference = from_time or datetime.now(self.tz)
        iterator = croniter(self.cron, reference)
        return iterator.get_next(datetime)


@dataclass(slots=True)
class WebhookConfig:
    """Configuration for webhook notifications."""

    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    payload_key: str = "message"
    timeout_seconds: float = 10.0


class WebhookNotifier:
    """Dispatches notifications via HTTP webhook calls."""

    def __init__(self, config: WebhookConfig) -> None:
        self.config = config

    def send(self, message: str) -> None:
        payload = {self.config.payload_key: message}
        try:
            response = httpx.request(
                self.config.method,
                self.config.url,
                headers=self.config.headers,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Webhook notification failed: %s", exc)
        else:
            LOGGER.debug(
                "Webhook notification dispatched (%s)",
                self.config.url,
            )


class SchedulerService:
    """Cron-style scheduler that executes actions using the registry."""

    def __init__(
        self,
        job: ScheduleJob,
        runner: ActionRunner,
        *,
        notifier: Optional[WebhookNotifier] = None,
    ) -> None:
        self.job = job
        self.runner = runner
        self.notifier = notifier
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._next_run: Optional[datetime] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            LOGGER.debug("SchedulerService already running")
            return
        self._stop_event.clear()
        self._next_run = self.job.next_run()
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()
        self._thread = thread
        LOGGER.info("SchedulerService started for action %s", self.job.action)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread:
            thread.join(timeout=5)
        LOGGER.info("SchedulerService stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now(self.job.tz)
            if self._next_run is None or now >= self._next_run:
                self._execute_job()
                self._next_run = self.job.next_run(from_time=now)
                continue
            sleep_seconds = (self._next_run - now).total_seconds()
            time.sleep(min(sleep_seconds, 60))

    def _execute_job(self) -> None:
        LOGGER.info("Executing scheduled action %s", self.job.action)
        if self.notifier:
            self.notifier.send(f"Running scheduled action {self.job.action}")
        try:
            self.runner(self.job.action, dict(self.job.inputs))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Scheduled job failed: %s", exc)
            if self.notifier:
                self.notifier.send(
                    f"Scheduled action {self.job.action} failed: {exc}"
                )
        else:
            if self.notifier:
                self.notifier.send(
                    f"Scheduled action {self.job.action} completed "
                    "successfully"
                )


__all__ = [
    "ScheduleJob",
    "SchedulerService",
    "WebhookConfig",
    "WebhookNotifier",
]
