"""Regression tests covering Phase 1 shell plumbing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llmscore.orchestrator.events import OrchestratorEvent
from llmscore.shell.preferences import (
    ShellPreferences,
    apply_preferences,
    load_preferences,
    save_preferences,
)
from llmscore.shell.suggestions import SuggestionEngine
from llmscore.shell.timeline import TimelineEntry, TimelineManager
from llmscore.state.policies import RetentionPolicy, SessionPolicyRegistry
from llmscore.state.store import SessionStore


@pytest.fixture()
def session_store(tmp_path):  # type: ignore[override]
    return SessionStore(path=tmp_path / "session.db")


def test_timeline_entry_preserves_event_metadata(session_store: SessionStore) -> None:
    event = OrchestratorEvent(
        kind="info",
        message="hello",
        timestamp=datetime.now(UTC),
        payload={"foo": "bar"},
        source="action",
        action="score.batch",
        checkpoint="init",
        run_id="run-123",
    )
    entry = TimelineEntry.from_event(event, session_id="alpha", profile="default")
    saved = session_store.save(entry.to_record())
    loaded = session_store.load(saved.identifier)
    assert loaded is not None
    restored = TimelineEntry.from_record(loaded)
    assert restored is not None
    assert restored.action == "score.batch"
    assert restored.source == "action"
    assert restored.checkpoint == "init"
    assert restored.run_id == "run-123"
    assert restored.payload == {"foo": "bar"}


def test_suggestion_engine_respects_policy_toggle(session_store: SessionStore) -> None:
    registry = SessionPolicyRegistry(
        {
            "suggestions": RetentionPolicy(category="suggestions", max_entries=0),
        }
    )
    session_store.set_policy_registry(registry)
    engine = SuggestionEngine(session_store=session_store)
    engine.bind_session("session-a", profile="ops")
    engine.record_action("foo")
    # Persistence disabled -> nothing stored
    assert session_store.list(category="suggestions") == []


def test_suggestion_engine_caps_recent_actions(session_store: SessionStore) -> None:
    registry = SessionPolicyRegistry(
        {
            "suggestions": RetentionPolicy(category="suggestions", max_entries=2),
        }
    )
    session_store.set_policy_registry(registry)
    engine = SuggestionEngine(session_store=session_store, recent_limit=5)
    engine.bind_session("session-b", profile="ops")
    for idx in range(5):
        engine.record_action(f"action-{idx}")
    record = session_store.load("ui.suggestions::session-b")
    assert record is not None
    recent = record.data.get("recent")
    assert isinstance(recent, list)
    assert len(recent) == 2
    assert recent[0].startswith("action-")


def test_config_preferences_persist_and_reload(session_store: SessionStore) -> None:
    prefs = ShellPreferences(
        retention_overrides={
            "timeline": {
                "ttl": "1d",
                "max_entries": 10,
            }
        }
    )
    save_preferences(session_store, profile="demo", preferences=prefs)
    reloaded = load_preferences(session_store, profile="demo")
    assert reloaded.retention_overrides == prefs.retention_overrides
    registry = apply_preferences(session_store, reloaded)
    resolved = registry.resolve("timeline")
    assert resolved.ttl is not None
    assert resolved.max_entries == 10


def test_timeline_manager_enforces_retention_policy(tmp_path) -> None:
    registry = SessionPolicyRegistry(
        {
            "timeline": RetentionPolicy(category="timeline", max_entries=2),
        }
    )
    store = SessionStore(path=tmp_path / "timeline.db", policies=registry)
    manager = TimelineManager(session_store=store)
    manager.bind_session("session-1", profile="demo")

    for idx in range(4):
        event = OrchestratorEvent(
            kind="info",
            message=f"event-{idx}",
            timestamp=datetime.now(UTC),
            source="action",
        )
        manager.append(event)

    records = store.list(category="timeline", include_expired=True)
    assert len(records) == 2

    # Rebind to ensure cache reloads from persisted data
    fresh_manager = TimelineManager(session_store=store)
    fresh_manager.bind_session("session-1", profile="demo")
    recent = list(fresh_manager.recent())
    assert [event.message for event in recent] == ["event-2", "event-3"]


def test_timeline_manager_rehydrates_metadata(tmp_path) -> None:
    store = SessionStore(path=tmp_path / "timeline_meta.db")
    manager = TimelineManager(session_store=store)
    manager.bind_session("sess", profile="demo")

    event = OrchestratorEvent(
        kind="info",
        message="metadata-test",
        timestamp=datetime.now(UTC),
        source="controller",
        action="score.batch",
        checkpoint="stage-1",
        run_id="run-xyz",
        payload={"foo": "bar"},
    )
    manager.append(event)

    reload = TimelineManager(session_store=store)
    reload.bind_session("sess", profile="demo")
    restored = list(reload.recent())
    assert len(restored) == 1
    assert restored[0].action == "score.batch"
    assert restored[0].source == "controller"
    assert restored[0].checkpoint == "stage-1"
    assert restored[0].payload == {"foo": "bar"}
