"""Phase 2 layout and dock scaffolding tests."""

from __future__ import annotations

from llmscore.shell.components import Dock, DockConfig
from llmscore.shell.layout import LayoutConfig, LayoutManager
from llmscore.shell.runtime import ShellConfig, ShellRuntime
from llmscore.state.store import SessionStore


def test_layout_manager_summary_reflects_visibility() -> None:
    manager = LayoutManager(config=LayoutConfig(enabled=True))
    manager.update_states(timeline=False, context=True, dock=False)
    summary = manager.summary()
    assert "Layout: experimental mode enabled" in summary
    assert "Timeline: hidden" in summary
    assert "Context: visible" in summary
    assert "Dock: hidden" in summary


def test_layout_manager_summary_legacy_mode() -> None:
    manager = LayoutManager(config=LayoutConfig(enabled=False))
    summary = manager.summary()
    assert "Layout: legacy" in summary
    assert "Timeline: visible" in summary
    assert "Context: visible" in summary
    assert "Dock: visible" in summary


def test_dock_snapshot_reports_status() -> None:
    dock = Dock(DockConfig(position="top", show_status_chip=True))
    dock.state.profile = "qa"
    dock.state.session_id = "abc123"
    dock.state.last_command = "run score.batch"
    dock.state.latency_ms = 42.0
    snapshot = dock.snapshot()
    assert "Dock position: top" in snapshot
    assert "Status chip: qa / abc123" in snapshot
    assert "Last command: run score.batch" in snapshot
    assert "Latency: 42.0 ms" in snapshot


def test_dock_snapshot_collapsed_without_status_chip() -> None:
    dock = Dock(DockConfig(position="bottom", show_status_chip=False))
    dock.toggle()
    snapshot = dock.snapshot()
    assert "Dock (collapsed)" not in snapshot  # only appears in render title
    assert "Dock position: bottom" in snapshot
    assert "Status chip" not in snapshot


def test_dock_toggle_updates_state() -> None:
    dock = Dock()
    assert dock.state.expanded is True
    dock.toggle()
    assert dock.state.expanded is False


def test_layout_manager_stream_limit_scales_with_frame() -> None:
    hits: list[str] = []
    manager = LayoutManager(
        config=LayoutConfig(enabled=True),
        command_stream=lambda: hits.append("command"),
        timeline=lambda: hits.append("timeline"),
        context=lambda: hits.append("context"),
        dock=lambda: hits.append("dock"),
    )
    manager.set_history_limit(50)
    manager.set_frame(120, 40)
    manager.render()
    assert manager.stream_limit == 30
    manager.set_frame(80, 8)
    manager.render()
    assert manager.stream_limit == 3
    assert hits  # render pipeline executed


def test_runtime_layout_flag_renders_without_error(tmp_path, monkeypatch) -> None:
    class _DummySession:
        def __init__(self, *args, **kwargs):
            self.history = kwargs.get("history")

        def prompt(self, *args, **kwargs):
            return ""

    monkeypatch.setattr("llmscore.shell.runtime.PromptSession", _DummySession)
    store = SessionStore(path=tmp_path / "runtime.db")
    config = ShellConfig(
        enable_layout_v2=True,
        session_id="demo",
        profile="qa",
    )
    runtime = ShellRuntime(config=config, session_store=store)
    runtime._render_status()
    runtime._record_command("help")
    runtime._render_layout_if_enabled()
    runtime.toggle_timeline()
    runtime.toggle_context()


def test_runtime_records_command_history(tmp_path, monkeypatch) -> None:
    class _DummySession:
        def __init__(self, *args, **kwargs):
            self.history = kwargs.get("history")

        def prompt(self, *args, **kwargs):
            return ""

    monkeypatch.setattr("llmscore.shell.runtime.PromptSession", _DummySession)
    store = SessionStore(path=tmp_path / "history.db")
    runtime = ShellRuntime(
        config=ShellConfig(enable_layout_v2=True, session_id="hist"),
        session_store=store,
    )
    for idx in range(205):
        runtime._record_command(f"cmd-{idx}")
    assert len(runtime._recent_commands) == runtime._command_history_limit
    assert runtime._recent_commands[-1] == "cmd-204"
    runtime._render_command_stream()


def test_runtime_toggle_invokes_layout_render(tmp_path, monkeypatch) -> None:
    class _DummySession:
        def __init__(self, *args, **kwargs):
            self.history = kwargs.get("history")

        def prompt(self, *args, **kwargs):  # pragma: no cover - stub
            return ""

    class _StubLayout:
        def __init__(
            self,
            *,
            config=None,
            command_stream=None,
            timeline=None,
            context=None,
            dock=None,
        ) -> None:
            self.config = config or LayoutConfig(enabled=True)
            self.render_calls = 0
            self.updated: list[dict[str, bool | None]] = []
            self._stream_limit = 5

        def set_history_limit(self, limit: int) -> None:  # noqa: D401 - stub
            self._stream_limit = min(self._stream_limit, limit)

        def update_states(
            self,
            *,
            timeline: bool | None = None,
            context: bool | None = None,
            dock: bool | None = None,
        ) -> None:
            self.updated.append({"timeline": timeline, "context": context, "dock": dock})

        def summary(self) -> str:
            return "stub summary"

        def set_frame(self, width: int, height: int) -> None:  # pragma: no cover - stub
            del width, height

        def render(self) -> None:
            self.render_calls += 1

        @property
        def stream_limit(self) -> int:
            return self._stream_limit

    monkeypatch.setattr("llmscore.shell.runtime.PromptSession", _DummySession)
    monkeypatch.setattr("llmscore.shell.runtime.LayoutManager", _StubLayout)
    store = SessionStore(path=tmp_path / "toggle.db")
    runtime = ShellRuntime(
        config=ShellConfig(enable_layout_v2=True, session_id="toggle"),
        session_store=store,
    )
    runtime.toggle_timeline()
    runtime.toggle_context()
    assert runtime.layout_manager.render_calls >= 1
    assert any(entry.get("timeline") is not None for entry in runtime.layout_manager.updated)
    assert any(entry.get("context") is not None for entry in runtime.layout_manager.updated)
