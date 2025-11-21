"""End-to-end runtime loop coverage for Phase 3 UX features."""

from __future__ import annotations

from types import SimpleNamespace

from llmscore.actions.base import ActionExecutionResult, ActionMetadata
from llmscore.shell.palette import PaletteEntry, PaletteProvider
from llmscore.shell.runtime import ShellConfig, ShellRuntime
from llmscore.state.store import SessionStore


class _StubActionRegistry:
    def __init__(self, results: dict[str, ActionExecutionResult]) -> None:
        self._results = results
        self._definitions = [SimpleNamespace(metadata=result.metadata) for result in results.values()]

    def bind_event_bus(self, _event_bus) -> None:  # pragma: no cover - wiring stub
        return None

    def run(self, name: str, *, inputs=None, controller=None) -> ActionExecutionResult:  # noqa: D401 - stub
        del inputs, controller
        return self._results[name]

    def __contains__(self, name: str) -> bool:
        return name in self._results

    def __iter__(self):  # pragma: no cover - palette bootstrap
        return iter(self._definitions)


def _patch_prompt_session(monkeypatch, commands: list[str]):
    created = []

    class _StubSession:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.kwargs = kwargs
            self.history = kwargs.get("history")
            self.default_buffer = SimpleNamespace(
                document=SimpleNamespace(text_before_cursor="")
            )
            self._commands = iter(commands)
            created.append(self)

        def prompt(self, *args, **kwargs):  # pragma: no cover - stub IO
            del args, kwargs
            try:
                return next(self._commands)
            except StopIteration as exc:  # ensure runtime stops cleanly if exhausted
                raise EOFError from exc

    monkeypatch.setattr("llmscore.shell.runtime.PromptSession", _StubSession)
    return created


class _StdoutPassthrough:
    def __enter__(self):  # pragma: no cover - context helper
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - context helper
        return False


def _make_result(name: str = "score.batch", *, duration: float = 0.12) -> ActionExecutionResult:
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


def test_runtime_run_loop_with_phase3_features(tmp_path, monkeypatch):
    commands = ["status", "run score.batch", "exit"]
    sessions = _patch_prompt_session(monkeypatch, commands)
    monkeypatch.setattr("llmscore.shell.runtime.patch_stdout", lambda: _StdoutPassthrough())
    monkeypatch.setattr(
        "llmscore.shell.runtime.FOLLOW_UP_HINTS",
        {
            "score.batch": (
                {
                    "command": "score.report",
                    "title": "Report",
                    "reason": "Summarize",
                },
                {
                    "command": "results.leaderboard",
                    "title": "Leaderboard",
                    "reason": "Compare",
                },
            )
        },
    )

    captured_card: dict[str, object] = {}

    class _FakeCard:
        @classmethod
        def from_payload(cls, **kwargs):
            captured_card["kwargs"] = kwargs
            return cls()

        def render(self, export_path=None):  # pragma: no cover - IO stub
            captured_card["render_called"] = True
            captured_card["export_path"] = export_path

    monkeypatch.setattr("llmscore.shell.runtime.OutputCard", _FakeCard)

    results = {"score.batch": _make_result()}
    palette = PaletteProvider(_palette_entries(), recent_limit=5)
    store = SessionStore(path=tmp_path / "runtime-e2e.db")
    runtime = ShellRuntime(
        config=ShellConfig(
            enable_layout_v2=True,
            use_output_cards=True,
            palette_autocomplete=True,
            session_id="e2e",
        ),
        action_registry=_StubActionRegistry(results),
        palette_provider=palette,
        session_store=store,
    )

    exit_code = runtime.run()

    assert exit_code == 0
    assert "run score.batch" in runtime._recent_commands
    assert runtime.dock.state.quick_actions  # populated via follow-up hints or recommendations
    assert captured_card.get("render_called") is True
    assert captured_card["kwargs"]["status"] == "success"
    assert sessions[-1].kwargs.get("bottom_toolbar") is not None
