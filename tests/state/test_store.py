"""Tests for SessionStore persistence behaviors."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from llmscore.state.policies import RetentionPolicy, SessionPolicyRegistry
from llmscore.state.store import SessionRecord, SessionStore


@pytest.fixture()
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(path=tmp_path / "session.db")


def test_save_infers_category_from_identifier(store: SessionStore) -> None:
    record = SessionRecord(identifier="timeline::abc", data={})
    saved = store.save(record)
    assert saved.category == "timeline"
    loaded = store.load("timeline::abc")
    assert loaded is not None
    assert loaded.category == "timeline"


def test_save_respects_explicit_category(store: SessionStore) -> None:
    record = SessionRecord(
        identifier="custom::123",
        data={},
        category="layout",
    )
    saved = store.save(record)
    assert saved.category == "layout"
    loaded = store.load("custom::123")
    assert loaded is not None
    assert loaded.category == "layout"


def test_expired_records_are_removed(tmp_path: Path) -> None:
    registry = SessionPolicyRegistry(
        {
            "general": RetentionPolicy(category="general", ttl=timedelta(seconds=0)),
        }
    )
    store = SessionStore(path=tmp_path / "expiry.db", policies=registry)
    record = SessionRecord(identifier="note::1", data={})
    saved = store.save(record)
    assert saved.expires_at is not None
    assert store.load("note::1") is None
    assert store.list(include_expired=False) == []
    assert store.list(include_expired=True)


def test_max_entries_policy_removes_old_records(tmp_path: Path) -> None:
    registry = SessionPolicyRegistry(
        {
            "general": RetentionPolicy(category="general", max_entries=1),
        }
    )
    store = SessionStore(path=tmp_path / "max.db", policies=registry)
    first = SessionRecord(identifier="note::1", data={})
    second = SessionRecord(identifier="note::2", data={})
    store.save(first)
    store.save(second)
    assert store.load("note::2") is not None
    assert store.load("note::1") is None


def test_list_filters_by_category(store: SessionStore) -> None:
    general_record = SessionRecord(identifier="task::1", data={})
    timeline_record = SessionRecord(identifier="timeline::1", data={})
    store.save(general_record)
    store.save(timeline_record)
    timeline_records = store.list(category="timeline")
    assert len(timeline_records) == 1
    assert timeline_records[0].identifier == "timeline::1"
