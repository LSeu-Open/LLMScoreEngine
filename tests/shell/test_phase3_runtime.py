"""Phase 3 runtime regressions covering palette/dock/output cards."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from llmscore.actions.base import ActionExecutionResult, ActionMetadata
from llmscore.shell.palette import PaletteCompleter, PaletteEntry, PaletteProvider
from llmscore.shell.runtime import ShellConfig, ShellRuntime
from llmscore.state.store import SessionStore


class _StubActionRegistry:
    def __init__(self, results: dict[str, ActionExecutionResult]) -> None:
        self._results = results

    def bind_event_bus(self, _event_bus) -> None:  # pragma: no cover - wiring stub
        return None

    def run(self, name: str, *, inputs=None, controller=None) -> ActionExecutionResult:  # noqa: D401 - stub
        del inputs, controller
        return self._results[name]

    def __contains__(self, name: str) -> bool:
        return name in self._results

    def __iter__(self):  # pragma: no cover - palette bootstrap stub
        return iter([])


def _patch_prompt_session(monkeypatch):
    created = []

    class _StubSession:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.kwargs = kwargs
            self.history = kwargs.get("history")
            self.default_buffer = SimpleNamespace(
                document=SimpleNamespace(text_before_cursor="")
            )
            created.append(self)

        def prompt(self, *args, **kwargs):  # pragma: no cover - unused in tests
            del args, kwargs
            return ""

    monkeypatch.setattr("llmscore.shell.runtime.PromptSession", _StubSession)
    return created


def _make_result(name: str = "score.batch", *, duration: float = 0.42) -> ActionExecutionResult:
    metadata = ActionMetadata(
        name=name,
        title=name.replace(".", " ").title(),
        description=f"Run {name}",
        domain="tests",
    )
    return ActionExecutionResult(metadata=metadata, output={"ok": True}, duration=duration)


def _palette_entries() -> list[PaletteEntry]:
    return [
        PaletteEntry(title="Score Batch", action="score.batch", description="Kick off scoring"),
        PaletteEntry(title="Score Report", action="score.report", description="Summarize batch"),
        PaletteEntry(title="Leaderboard", action="results.leaderboard", description="Compare runs"),
    ]


def _build_runtime(
    tmp_path,
    monkeypatch,
    *,
    config: ShellConfig,
    results: dict[str, ActionExecutionResult],
    store_name: str,
):
    sessions = _patch_prompt_session(monkeypatch)
    palette = PaletteProvider(_palette_entries(), recent_limit=5)
    store = SessionStore(path=tmp_path / store_name)
    runtime = ShellRuntime(
        config=config,
        action_registry=_StubActionRegistry(results),
        palette_provider=palette,
        session_store=store,
    )
    return runtime, sessions[-1]


@pytest.mark.parametrize("flag_enabled", [True, False])
def test_palette_autocomplete_flag_controls_toolbar(tmp_path, monkeypatch, flag_enabled):
    results = {"score.batch": _make_result()}
    config = ShellConfig(
        palette_autocomplete=flag_enabled,
        use_output_cards=False,
        session_id="palette-toggle" if flag_enabled else "palette-off",
    )
    runtime, session = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name=("palette-on.db" if flag_enabled else "palette-off.db"),
    )
    if flag_enabled:
        assert isinstance(runtime._completer, PaletteCompleter)
        assert "bottom_toolbar" in session.kwargs
        assert "completer" in session.kwargs
    else:
        assert runtime._completer is None
        assert "bottom_toolbar" not in session.kwargs
        assert "completer" not in session.kwargs


def test_quick_actions_refresh_after_command(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llmscore.shell.runtime.FOLLOW_UP_HINTS",
        {
            "score.batch": (
                {"command": "rerun.score", "title": "Rerun", "reason": "Retry"},
                {"command": "open.timeline", "title": "Timeline", "reason": "Inspect"},
            )
        },
    )
    results = {"score.batch": _make_result(duration=0.1)}
    config = ShellConfig(palette_autocomplete=False, session_id="quick-actions")
    runtime, _ = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name="quick-actions.db",
    )
    runtime._execute_action("score.batch")
    quick = runtime.dock.state.quick_actions
    assert quick[:2] == ["rerun.score", "open.timeline"]
    assert len(quick) <= 3


def test_output_card_render_path(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class _FakeCard:
        @classmethod
        def from_payload(cls, **kwargs):
            captured["kwargs"] = kwargs
            return cls()

        def render(self, export_path=None):  # pragma: no cover - simple stub
            captured["render_called"] = True
            captured["export_path"] = export_path

    monkeypatch.setattr("llmscore.shell.runtime.OutputCard", _FakeCard)
    results = {"score.batch": _make_result(duration=0.33)}
    config = ShellConfig(
        palette_autocomplete=False,
        use_output_cards=True,
        session_id="output-card",
    )
    runtime, _ = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name="output-card.db",
    )
    runtime._execute_action("score.batch")
    assert captured.get("render_called") is True
    card_kwargs = captured["kwargs"]
    assert card_kwargs["status"] == "success"
    assert card_kwargs["metadata"]["Action"] == "score.batch"
    assert card_kwargs["metadata"]["Duration"].endswith("s")


def test_timeline_respects_reduced_motion(tmp_path, monkeypatch):
    captured: dict[str, str] = {}

    def _capture_panel(content, title=None, options=None):  # noqa: D401 - stub
        captured["content"] = content
        captured["title"] = title

    monkeypatch.setattr(
        "llmscore.shell.runtime.render_panel",
        _capture_panel,
    )
    results = {"score.batch": _make_result()}
    config = ShellConfig(reduced_motion=True, session_id="timeline-reduced")
    runtime, _ = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name="timeline-reduced.db",
    )
    runtime.timeline.recent = lambda limit: [
        SimpleNamespace(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            kind="info",
            message="Reduced motion event",
        )
    ]
    runtime._render_timeline_snapshot()
    assert "Reduced motion event" in captured["content"]
    assert captured["title"] == "Timeline"


def test_color_blind_palette_passed_to_tables(tmp_path, monkeypatch):
    recorded: dict[str, object] = {}

    def _capture_style(table, *, options=None):  # noqa: D401 - stub
        recorded["options"] = options

    monkeypatch.setattr(
        "llmscore.shell.runtime.style_table",
        _capture_style,
    )
    results = {"score.batch": _make_result()}
    config = ShellConfig(color_blind_mode=True, session_id="colorblind")
    runtime, _ = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name="colorblind.db",
    )
    runtime._recent_commands = ["cmd-1", "cmd-2"]
    runtime._render_command_stream()
    palette = recorded["options"].palette
    assert palette.panel_border == "bright_white"
    assert palette.success == "bright_cyan"


def test_focus_cycle_renders_visible_panes(tmp_path, monkeypatch):
    results = {"score.batch": _make_result()}
    config = ShellConfig(session_id="focus-cycle")
    runtime, _ = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name="focus-cycle.db",
    )
    monkeypatch.setattr(runtime.console, "print", lambda *args, **kwargs: None)
    calls: list[str] = []
    runtime._render_timeline_snapshot = lambda: calls.append("timeline")  # type: ignore[assignment]
    runtime._render_context = lambda: calls.append("context")  # type: ignore[assignment]
    runtime._render_dock = lambda: calls.append("dock")  # type: ignore[assignment]
    runtime._cycle_focus()  # context
    runtime._cycle_focus()  # command (no render)
    runtime._cycle_focus()  # dock
    runtime._cycle_focus()  # timeline
    assert calls == ["context", "dock", "timeline"]


def test_toggle_context_updates_layout_and_renders(tmp_path, monkeypatch):
    results = {"score.batch": _make_result()}
    config = ShellConfig(session_id="toggle-context")
    runtime, _ = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name="toggle-context.db",
    )
    monkeypatch.setattr(runtime.console, "print", lambda *args, **kwargs: None)
    context_states: list[bool | None] = []
    runtime.layout_manager.update_states = (  # type: ignore[assignment]
        lambda **kwargs: context_states.append(kwargs.get("context"))
    )
    renders: list[str] = []
    runtime._render_context = lambda: renders.append("context")  # type: ignore[assignment]
    runtime.context_visible = False
    runtime.toggle_context()
    assert runtime.context_visible is True
    assert context_states[-1] is True
    assert renders == ["context"]
    runtime.toggle_context()
    assert runtime.context_visible is False
    assert context_states[-1] is False


def test_performance_monitor_collects_samples(tmp_path, monkeypatch):
    results = {"score.batch": _make_result()}
    config = ShellConfig(
        session_id="perf-samples",
        performance_monitor_enabled=True,
    )
    runtime, _ = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name="perf-samples.db",
    )
    runtime._render_command_stream()
    summary = runtime.performance_summary()
    assert "command_stream_render" in summary
    assert summary["command_stream_render"]["budget_ms"] is not None


def test_performance_command_renders_panel(tmp_path, monkeypatch):
    recorded: dict[str, object] = {}

    def _capture_panel(content, title=None, options=None):  # noqa: D401 - stub
        recorded["content"] = content
        recorded["title"] = title

    monkeypatch.setattr("llmscore.shell.runtime.render_panel", _capture_panel)
    results = {"score.batch": _make_result()}
    config = ShellConfig(
        session_id="perf-command",
        performance_monitor_enabled=True,
    )
    runtime, _ = _build_runtime(
        tmp_path,
        monkeypatch,
        config=config,
        results=results,
        store_name="perf-command.db",
    )
    runtime._render_command_stream()
    runtime._dispatch_command("performance")
    assert recorded["title"] == "Performance"
    assert "command_stream_render" in recorded["content"]
