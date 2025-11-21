"""Scripted shell transcripts for wizard-style journeys."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from llmscore.actions.base import ActionExecutionResult, ActionMetadata
from llmscore.orchestrator.events import OrchestratorEvent
from llmscore.shell.palette import PaletteEntry
from llmscore.shell.runtime import ShellConfig, ShellRuntime
from llmscore.state.store import SessionStore

from .recorder import TranscriptEvent, TranscriptRecorder

BASELINES_DIR = Path(__file__).with_suffix("").parent / "baselines"


@dataclass(slots=True)
class _StubActionRegistry:
    results: dict[str, list[ActionExecutionResult]]
    progress_script: dict[str, list[list[dict[str, object]]]] | None = None
    progress_counts: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def bind_event_bus(self, _event_bus) -> None:  # pragma: no cover - wiring stub
        return None

    def run(self, name: str, *, controller=None, inputs=None) -> ActionExecutionResult:  # noqa: D401 - stub call
        del controller, inputs
        queue = self.results[name]
        result = queue.pop(0)
        queue.append(result)
        return result

    def __contains__(self, name: str) -> bool:
        return name in self.results

    def __iter__(self):  # pragma: no cover - palette bootstrap helper
        for queue in self.results.values():
            yield SimpleNamespace(metadata=queue[0].metadata)


_STUB_PROGRESS_PATCHED = False
_ORIGINAL_STUB_RUN = _StubActionRegistry.run


def _ensure_progress_patch() -> None:
    global _STUB_PROGRESS_PATCHED
    if _STUB_PROGRESS_PATCHED:
        return

    def _run_with_progress(self, name: str, *, controller=None, inputs=None):  # type: ignore[override]
        script = getattr(self, "progress_script", None)
        if controller and script:
            sequences = script.get(name)
            if sequences:
                counts = getattr(self, "progress_counts", None)
                if counts is None:
                    counts = defaultdict(int)
                    setattr(self, "progress_counts", counts)
                index = min(counts[name], len(sequences) - 1)
                for step in sequences[index]:
                    payload = {
                        key: value
                        for key, value in step.items()
                        if key != "message" and value is not None
                    }
                    controller.record_event(
                        OrchestratorEvent(
                            kind="progress",
                            message=step["message"],
                            timestamp=datetime.now(UTC),
                            payload=payload or None,
                            action=name,
                        )
                    )
                counts[name] += 1
        return _ORIGINAL_STUB_RUN(self, name, controller=controller, inputs=inputs)

    _StubActionRegistry.run = _run_with_progress  # type: ignore[assignment]
    _STUB_PROGRESS_PATCHED = True


def _make_result(name: str, payload: dict) -> ActionExecutionResult:
    return ActionExecutionResult(
        metadata=ActionMetadata(
            name=name,
            title=name.title(),
            description=f"Wizard flow for {name}",
            domain="tests",
        ),
        output=payload,
        duration=0.01,
    )


@dataclass(slots=True)
class _StubPaletteProvider:
    entries: list[PaletteEntry]

    def record_usage(self, action: str) -> None:  # pragma: no cover - optional
        return None

    def search(self, query: str, *, limit: int = 5):
        query = query.lower().strip()
        matching = [entry for entry in self.entries if query in entry.action.lower() or query in entry.title.lower()]
        yield from matching[:limit] if matching else self.recommend(limit=limit)

    def recommend(self, *, limit: int = 5):
        yield from self.entries[:limit]

    def recent(self, *, limit: int = 5):  # pragma: no cover - not used
        yield from self.entries[:limit]

    def all_entries(self):  # pragma: no cover
        return list(self.entries)

    def __iter__(self):
        return iter(self.entries)


class _NoopContext:
    def __enter__(self):  # pragma: no cover - helper
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - helper
        return False


class _BufferConsole:
    def __init__(self) -> None:
        self._chunks: list[str] = []
        self.width = 100
        self.height = 30

    @staticmethod
    def _normalize(arg: object) -> str:
        if hasattr(arg, "__class__"):
            module = getattr(arg.__class__, "__module__", "")
            if module.startswith("rich"):
                return f"<{arg.__class__.__name__}>"
        return str(arg)

    def print(self, *args, **kwargs) -> None:  # pragma: no cover - IO shim
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(self._normalize(arg) for arg in args) + end
        self._chunks.append(text)

    def rule(self, text: str) -> None:
        self._chunks.append(f"=== {text} ===\n")

    def clear(self) -> None:
        self._chunks.append("[clear]\n")

    def getvalue(self) -> str:
        return "".join(self._chunks)


class _ScriptedSession:
    def __init__(self, harness: "_ScriptedShellHarness") -> None:
        self._harness = harness
        self.default_buffer = SimpleNamespace(
            document=SimpleNamespace(text_before_cursor="")
        )

    def prompt(self, *_, **__):  # pragma: no cover - stubbed prompt
        # Flush the previous command before accepting the next one so outputs
        # stay aligned with prompts
        self._harness.flush_step()
        return self._harness.next_command()


class _ScriptedShellHarness:
    def __init__(self, commands: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
        self._commands = list(commands)
        self.console = _BufferConsole()
        self._active_command: str | None = None
        self._offset = 0
        self.events: list[TranscriptEvent] = []
        self._timestamp_pattern = re.compile(r"\[\d{2}:\d{2}:\d{2}\]")

        def fake_get_console(*_args, **_kwargs):
            return self.console

        def fake_render_panel(content, *, title: str | None = None, **_kwargs):
            header = f"[{title}]\n" if title else ""
            fake_get_console()
            self.console.print(header + str(content).strip())

        monkeypatch.setattr("llmscore.utils.rich_render.get_console", fake_get_console)
        monkeypatch.setattr("llmscore.shell.runtime.get_console", fake_get_console)
        monkeypatch.setattr("llmscore.shell.layout.get_console", fake_get_console)
        monkeypatch.setattr("llmscore.utils.rich_render.render_panel", fake_render_panel)
        monkeypatch.setattr("llmscore.shell.runtime.render_panel", fake_render_panel)
        monkeypatch.setattr("llmscore.utils.rich_render.style_table", lambda *a, **k: None)
        class _FakeCard:  # pragma: no cover - deterministic output card helper
            def __init__(self, payload):
                self.payload = payload

            @classmethod
            def from_payload(cls, **kwargs):
                return cls(kwargs)

            def render(self, export_path=None):
                fake_get_console().print("[OutputCard]")
                fake_get_console().print(str(self.payload))

        monkeypatch.setattr("llmscore.shell.runtime.OutputCard", _FakeCard)
        monkeypatch.setattr("llmscore.shell.runtime.patch_stdout", _NoopContext)
        monkeypatch.setattr("llmscore.shell.runtime.PromptSession", lambda *_, **__: _ScriptedSession(self))

    def flush_step(self) -> None:
        if self._active_command is None:
            return
        content = self.console.getvalue()
        delta = content[self._offset :]
        self._offset = len(content)
        text = delta.strip() or "(no output)"
        text = self._timestamp_pattern.sub("[HH:MM:SS]", text)
        self.events.append(
            TranscriptEvent(prompt=f"shell> {self._active_command}", response=text)
        )
        self._active_command = None

    def next_command(self) -> str:
        if not self._commands:
            raise EOFError
        command = self._commands.pop(0)
        self._active_command = command
        return command

    def run(self, runtime: ShellRuntime) -> list[TranscriptEvent]:
        try:
            runtime.run()
        except EOFError:
            pass
        finally:
            self.flush_step()
        return self.events


def _make_result(name: str, payload: dict) -> ActionExecutionResult:
    return ActionExecutionResult(
        metadata=ActionMetadata(
            name=name,
            title=name.title(),
            description=f"Wizard flow for {name}",
            domain="tests",
        ),
        output=payload,
        duration=0.01,
    )


def _transcript_recorder() -> TranscriptRecorder:
    return TranscriptRecorder(BASELINES_DIR)


def _build_runtime(
    monkeypatch,
    tmp_path,
    commands: list[str],
    actions: dict[str, dict],
    *,
    palette_entries: list[PaletteEntry] | None = None,
    config_overrides: dict | None = None,
) -> tuple[ShellRuntime, _ScriptedShellHarness]:
    harness = _ScriptedShellHarness(commands, monkeypatch)
    prepared_results: dict[str, list[ActionExecutionResult]] = {}
    for name, payload in actions.items():
        payloads = payload if isinstance(payload, list) else [payload]
        prepared_results[name] = [_make_result(name, item) for item in payloads]
    registry = _StubActionRegistry(prepared_results)
    default_palette = [
        PaletteEntry(title="Score Model", action="score.model", description="Run model wizard"),
        PaletteEntry(title="Leaderboard", action="results.leaderboard", description="View leaderboard"),
        PaletteEntry(title="Compare", action="results.compare", description="Compare runs"),
        PaletteEntry(title="Automation Watch", action="automation.watch", description="Start watch job"),
        PaletteEntry(title="Automation List", action="automation.list", description="List jobs"),
        PaletteEntry(title="Automation Stop", action="automation.stop", description="Stop watch jobs"),
        PaletteEntry(title="Analyze Failures", action="results.analyze_failures", description="Drill into failures"),
        PaletteEntry(title="Pin Results", action="results.pin", description="Pin leaderboard entries"),
        PaletteEntry(title="Performance Overlay", action="results.performance", description="Show performance overlays"),
        PaletteEntry(title="Data Template", action="data.template", description="Bootstrap model template"),
        PaletteEntry(title="Data Fill", action="data.fill", description="Queue data fill"),
    ]
    palette = _StubPaletteProvider(list(palette_entries or default_palette))
    config_kwargs = dict(
        enable_layout_v2=True,
        use_output_cards=True,
        session_id="wizard",
    )
    if config_overrides:
        config_kwargs.update(config_overrides)
    runtime = ShellRuntime(
        config=ShellConfig(**config_kwargs),
        action_registry=registry,
        palette_provider=palette,
        session_store=SessionStore(path=tmp_path / "shell.db"),
    )
    original_dispatch = runtime._dispatch_command

    def _wrapped_dispatch(self, command: str):
        result = original_dispatch(command)
        harness.flush_step()
        return result

    runtime._dispatch_command = MethodType(_wrapped_dispatch, runtime)
    return runtime, harness


def _install_progress_script(
    runtime: ShellRuntime,
    progress_script: dict[str, list[list[dict[str, object]]]],
) -> None:
    if not isinstance(runtime.action_registry, _StubActionRegistry):
        return
    runtime.action_registry.progress_script = progress_script
    runtime.action_registry.progress_counts = defaultdict(int)
    _ensure_progress_patch()


def _record_shell_transcript(flow_name: str, events: list[TranscriptEvent]) -> None:
    _transcript_recorder().assert_match(flow_name, events)


def test_shell_wizard_onboarding_transcript(monkeypatch, tmp_path):
    commands = [
        "help",
        "palette score",
        "run score.model",
        "exit",
    ]
    actions = {
        "score.model": {
            "status": "ok",
            "model": "gamma",
            "score": {"final": 0.91},
        }
    }
    runtime, harness = _build_runtime(monkeypatch, tmp_path, commands, actions)
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_onboarding", events)


def test_shell_wizard_analytics_transcript(monkeypatch, tmp_path):
    commands = [
        "status",
        "palette results",
        "run results.leaderboard",
        "run results.compare",
        "exit",
    ]
    actions = {
        "results.leaderboard": {
            "entries": [
                {"model": "alpha", "score": 0.95},
                {"model": "beta", "score": 0.90},
            ]
        },
        "results.compare": {
            "primary": "alpha",
            "secondary": "beta",
            "delta": {"final_score": 0.05},
        },
    }
    runtime, harness = _build_runtime(monkeypatch, tmp_path, commands, actions)
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_analytics", events)


def test_shell_wizard_automation_transcript(monkeypatch, tmp_path):
    commands = [
        "timeline",
        "context",
        "run automation.watch",
        "run automation.list",
        "exit",
    ]
    actions = {
        "automation.watch": {
            "job_id": "watch-models",
            "paths": ["models/"],
            "action": "score.batch",
        },
        "automation.list": {
            "jobs": [
                {
                    "id": "watch-models",
                    "action": "score.batch",
                    "paths": ["models/"],
                }
            ]
        },
    }
    runtime, harness = _build_runtime(monkeypatch, tmp_path, commands, actions)
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_automation", events)


def test_shell_wizard_palette_onboarding_variant_transcript(monkeypatch, tmp_path):
    commands = [
        "palette",
        "palette data",
        "run data.template",
        "run data.fill",
        "run score.model",
        "exit",
    ]
    actions = {
        "data.template": {
            "model": "gamma",
            "path": "models/gamma.json",
            "status": "created",
        },
        "data.fill": {
            "model": "gamma",
            "status": "queued",
            "tasks": 4,
        },
        "score.model": {
            "status": "ok",
            "model": "gamma",
            "score": {"final": 0.92},
        },
    }
    palette_entries = [
        PaletteEntry(title="Data Template", action="data.template", description="Start template"),
        PaletteEntry(title="Data Fill", action="data.fill", description="Queue fill"),
        PaletteEntry(title="Score Model", action="score.model", description="Run scoring"),
    ]
    runtime, harness = _build_runtime(
        monkeypatch,
        tmp_path,
        commands,
        actions,
        palette_entries=palette_entries,
    )
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_palette_onboarding_variant", events)


def test_shell_wizard_analytics_drilldown_transcript(monkeypatch, tmp_path):
    commands = [
        "status",
        "palette results",
        "run results.leaderboard",
        "run results.analyze_failures",
        "run results.compare",
        "exit",
    ]
    actions = {
        "results.leaderboard": {
            "entries": [
                {"model": "alpha", "score": 0.96},
                {"model": "beta", "score": 0.91},
            ],
            "sort_key": "score",
        },
        "results.analyze_failures": {
            "failures": [
                {"model": "beta", "count": 3, "top_issue": "timeout"},
            ],
            "drilldowns": [
                {"model": "beta", "sample": "logs/beta.txt"},
            ],
        },
        "results.compare": {
            "primary": "alpha",
            "secondary": "beta",
            "delta": {"final_score": 0.05},
        },
    }
    palette_entries = [
        PaletteEntry(title="Leaderboard", action="results.leaderboard", description="View leaderboard"),
        PaletteEntry(title="Compare", action="results.compare", description="Compare runs"),
        PaletteEntry(title="Analyze Failures", action="results.analyze_failures", description="Drill into failures"),
    ]
    runtime, harness = _build_runtime(
        monkeypatch,
        tmp_path,
        commands,
        actions,
        palette_entries=palette_entries,
    )
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_analytics_drilldown", events)


def test_shell_wizard_automation_stop_list_transcript(monkeypatch, tmp_path):
    commands = [
        "palette automation",
        "run automation.list",
        "run automation.stop",
        "exit",
    ]
    actions = {
        "automation.list": {
            "jobs": [
                {
                    "id": "watch-models",
                    "action": "score.batch",
                    "paths": ["models/"],
                    "status": "active",
                },
                {
                    "id": "watch-datasets",
                    "action": "score.batch",
                    "paths": ["datasets/"],
                    "status": "inactive",
                },
            ]
        },
        "automation.stop": {
            "job_id": "watch-models",
            "status": "stopped",
            "message": "Watch stopped",
        },
    }
    palette_entries = [
        PaletteEntry(title="Automation List", action="automation.list", description="List jobs"),
        PaletteEntry(title="Automation Stop", action="automation.stop", description="Stop job"),
        PaletteEntry(title="Automation Watch", action="automation.watch", description="Start watch job"),
    ]
    runtime, harness = _build_runtime(
        monkeypatch,
        tmp_path,
        commands,
        actions,
        palette_entries=palette_entries,
    )
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_automation_stop_list", events)


def test_shell_wizard_automation_watch_progress_transcript(monkeypatch, tmp_path):
    commands = [
        "palette automation",
        "run automation.watch",
        "run automation.watch",
        "run automation.list",
        "run automation.stop",
        "run automation.list",
        "exit",
    ]
    actions = {
        "automation.watch": [
            {
                "job_id": "watch-models",
                "action": "score.batch",
                "paths": ["models/"],
                "status": "starting",
                "progress": {"percent": 25, "description": "Queued files"},
            },
            {
                "job_id": "watch-models",
                "action": "score.batch",
                "paths": ["models/"],
                "status": "running",
                "progress": {"percent": 80, "description": "Scoring"},
            },
        ],
        "automation.list": [
            {
                "jobs": [
                    {
                        "id": "watch-models",
                        "action": "score.batch",
                        "paths": ["models/"],
                        "status": "running",
                        "progress": 0.8,
                    }
                ]
            },
            {
                "jobs": [
                    {
                        "id": "watch-models",
                        "action": "score.batch",
                        "paths": ["models/"],
                        "status": "error",
                        "message": "Stop failed",
                    }
                ]
            },
        ],
        "automation.stop": {
            "job_id": "watch-models",
            "status": "error",
            "message": "Job not found on worker",
        },
    }
    palette_entries = [
        PaletteEntry(title="Automation Watch", action="automation.watch", description="Start watch job"),
        PaletteEntry(title="Automation List", action="automation.list", description="List jobs"),
        PaletteEntry(title="Automation Stop", action="automation.stop", description="Stop watch jobs"),
    ]
    runtime, harness = _build_runtime(
        monkeypatch,
        tmp_path,
        commands,
        actions,
        palette_entries=palette_entries,
    )
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_automation_watch_progress", events)


def test_shell_wizard_automation_long_lived_progress_transcript(monkeypatch, tmp_path):
    commands = [
        "palette automation",
        "run automation.watch",
        "run automation.list",
        "run automation.watch",
        "run automation.list",
        "exit",
    ]
    actions = {
        "automation.watch": [
            {
                "job_id": "watch-models",
                "action": "score.batch",
                "paths": ["models/"],
                "status": "starting",
                "progress": {"percent": 12, "description": "Initializing"},
            },
            {
                "job_id": "watch-models",
                "action": "score.batch",
                "paths": ["models/"],
                "status": "running",
                "progress": {"percent": 88, "description": "Streaming events"},
            },
        ],
        "automation.list": [
            {
                "jobs": [
                    {
                        "id": "watch-models",
                        "action": "score.batch",
                        "paths": ["models/"],
                        "status": "running",
                        "progress": 0.6,
                    }
                ]
            },
            {
                "jobs": [
                    {
                        "id": "watch-models",
                        "action": "score.batch",
                        "paths": ["models/"],
                        "status": "stabilizing",
                        "progress": 0.95,
                    }
                ]
            },
        ],
    }
    palette_entries = [
        PaletteEntry(title="Automation Watch", action="automation.watch", description="Start watch job"),
        PaletteEntry(title="Automation List", action="automation.list", description="List jobs"),
    ]
    runtime, harness = _build_runtime(
        monkeypatch,
        tmp_path,
        commands,
        actions,
        palette_entries=palette_entries,
    )
    _install_progress_script(
        runtime,
        {
            "automation.watch": [
                [
                    {"message": "watch-models: enqueue detected files (5%)", "percent": 5},
                    {"message": "watch-models: hashing assets (15%)", "percent": 15},
                    {"message": "watch-models: awaiting workers (35%)", "percent": 35},
                    {"message": "watch-models: dispatching batch (55%)", "percent": 55},
                ],
                [
                    {"message": "watch-models: processing backlog (72%)", "percent": 72},
                    {"message": "watch-models: applying throttling (85%)", "percent": 85},
                    {"message": "watch-models: finalizing checkpoints (96%)", "percent": 96},
                ],
            ]
        },
    )
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_automation_long_lived_progress", events)


def test_shell_wizard_analytics_pin_export_transcript(monkeypatch, tmp_path):
    commands = [
        "status",
        "palette results",
        "run results.leaderboard",
        "run results.analyze_failures",
        "run results.pin",
        "run results.performance",
        "exit",
    ]
    actions = {
        "results.leaderboard": {
            "entries": [
                {"model": "alpha", "score": 0.97},
                {"model": "beta", "score": 0.92},
                {"model": "gamma", "score": 0.89},
            ],
            "sort_key": "score",
        },
        "results.analyze_failures": {
            "exports": [
                {"artifact": "alpha_failures.csv", "status": "ready"},
                {"artifact": "beta_failures.csv", "status": "ready"},
            ],
            "clusters": [
                {"label": "timeout", "count": 2},
                {"label": "memory", "count": 1},
            ],
        },
        "results.pin": {
            "pinned": [
                {"model": "alpha", "score": 0.97},
                {"model": "beta", "score": 0.92},
            ],
            "note": "Pinned top performers",
        },
        "results.performance": {
            "timeline": [
                {"ts": "2024-01-01T00:00:00Z", "throughput": 42},
                {"ts": "2024-01-01T01:00:00Z", "throughput": 40},
            ],
            "budget": {"p95": 4.2, "target": 5.0},
        },
    }
    palette_entries = [
        PaletteEntry(title="Leaderboard", action="results.leaderboard", description="View leaderboard"),
        PaletteEntry(title="Analyze Failures", action="results.analyze_failures", description="Drill into failures"),
        PaletteEntry(title="Pin Results", action="results.pin", description="Pin leaderboard entries"),
        PaletteEntry(title="Performance Overlay", action="results.performance", description="Show performance overlays"),
    ]
    runtime, harness = _build_runtime(
        monkeypatch,
        tmp_path,
        commands,
        actions,
        palette_entries=palette_entries,
    )
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_analytics_pin_export", events)


def test_shell_wizard_onboarding_reduced_motion_transcript(monkeypatch, tmp_path):
    commands = [
        "help",
        "run score.model",
        "exit",
    ]
    actions = {
        "score.model": {
            "status": "ok",
            "model": "gamma",
            "score": {"final": 0.9},
        }
    }
    runtime, harness = _build_runtime(
        monkeypatch,
        tmp_path,
        commands,
        actions,
        config_overrides={"reduced_motion": True},
    )
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_onboarding_reduced_motion", events)


def test_shell_wizard_analytics_color_blind_transcript(monkeypatch, tmp_path):
    commands = [
        "status",
        "run results.leaderboard",
        "exit",
    ]
    actions = {
        "results.leaderboard": {
            "entries": [
                {"model": "alpha", "score": 0.95},
                {"model": "beta", "score": 0.91},
            ],
            "sort_key": "score",
        }
    }
    runtime, harness = _build_runtime(
        monkeypatch,
        tmp_path,
        commands,
        actions,
        config_overrides={"color_blind_mode": True},
    )
    events = harness.run(runtime)
    _record_shell_transcript("shell_wizard_analytics_color_blind", events)
