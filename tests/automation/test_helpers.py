from datetime import datetime, timezone
from pathlib import Path

import pytest

from llmscore.automation.scheduler import (
    ScheduleJob,
    WebhookConfig,
    WebhookNotifier,
)
from llmscore.automation.watchers import build_watch_job


def test_build_watch_job_creates_targets(tmp_path: Path) -> None:
    job = build_watch_job(
        [tmp_path],
        action="score.batch",
        patterns=["*.json"],
        ignore_patterns=["*.tmp"],
        debounce_seconds=2.5,
    )

    assert job.action == "score.batch"
    assert len(job.targets) == 1
    target = job.targets[0]
    assert target.normalized_path() == tmp_path.resolve()
    assert target.patterns == ("*.json",)
    assert target.ignore_patterns == ("*.tmp",)
    assert pytest.approx(job.debounce_seconds, rel=1e-6) == 2.5


def test_schedule_job_next_run_advances() -> None:
    job = ScheduleJob(action="automation", cron="*/5 * * * *")
    reference = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    next_run = job.next_run(from_time=reference)

    assert next_run.tzinfo == timezone.utc
    assert next_run > reference
    assert next_run.minute % 5 == 0


def test_webhook_notifier_send(monkeypatch) -> None:
    calls = []

    def fake_request(method, url, headers=None, json=None, timeout=None):
        class _Response:
            def raise_for_status(self):
                return None

        calls.append((method, url, headers, json, timeout))
        return _Response()

    monkeypatch.setattr(
        "llmscore.automation.scheduler.httpx.request",
        fake_request,
    )

    config = WebhookConfig(
        url="https://example.com/hook",
        method="POST",
        headers={"X-Test": "1"},
        payload_key="payload",
        timeout_seconds=3.0,
    )
    notifier = WebhookNotifier(config)

    notifier.send("Hello world")

    assert len(calls) == 1
    method, url, headers, payload, timeout = calls[0]
    assert method == "POST"
    assert url == "https://example.com/hook"
    assert headers == {"X-Test": "1"}
    assert payload == {"payload": "Hello world"}
    assert timeout == 3.0
