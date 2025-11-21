"""Timeline persistence helpers for the interactive shell."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

from ..orchestrator.events import OrchestratorEvent
from ..state.store import SessionRecord, SessionStore
from ..utils.logging import get_logger

LOGGER = get_logger("llmscore.shell.timeline")


@dataclass(slots=True)
class TimelineEntry:
    """Serialized timeline entry used for persistence and replay."""

    identifier: str
    session_id: str
    profile: Optional[str]
    timestamp: datetime
    kind: str
    message: str
    source: str
    action: Optional[str]
    checkpoint: Optional[str]
    run_id: Optional[str]
    payload: Optional[dict]

    @classmethod
    def from_event(
        cls,
        event: OrchestratorEvent,
        *,
        session_id: str,
        profile: Optional[str],
    ) -> "TimelineEntry":
        return cls(
            identifier=_event_record_id(session_id, event.id),
            session_id=session_id,
            profile=profile,
            timestamp=event.timestamp,
            kind=event.kind,
            message=event.message,
            source=event.source,
            action=event.action,
            checkpoint=event.checkpoint,
            run_id=event.run_id,
            payload=dict(event.payload or {}),
        )

    def to_event(self) -> OrchestratorEvent:
        return OrchestratorEvent(
            id=self.identifier.split("::")[-1],
            kind=self.kind,  # type: ignore[arg-type]
            message=self.message,
            timestamp=self.timestamp,
            payload=self.payload,
            source=self.source,  # type: ignore[arg-type]
            action=self.action,
            checkpoint=self.checkpoint,
            run_id=self.run_id,
        )

    def to_record(self) -> SessionRecord:
        return SessionRecord(
            identifier=self.identifier,
            profile=self.profile,
            data={
                "session_id": self.session_id,
                "timestamp": self.timestamp.isoformat(),
                "kind": self.kind,
                "message": self.message,
                "source": self.source,
                "action": self.action,
                "checkpoint": self.checkpoint,
                "run_id": self.run_id,
                "payload": self.payload,
            },
        )

    @classmethod
    def from_record(cls, record: SessionRecord) -> Optional["TimelineEntry"]:
        if not isinstance(record.data, dict):
            return None
        session_id = record.data.get("session_id")
        ts_text = record.data.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(ts_text) if isinstance(ts_text, str) else None
        except ValueError:
            timestamp = None
        if not isinstance(session_id, str) or timestamp is None:
            return None
        return cls(
            identifier=record.identifier,
            session_id=session_id,
            profile=record.profile,
            timestamp=timestamp,
            kind=str(record.data.get("kind", "info")),
            message=str(record.data.get("message", "")),
            source=str(record.data.get("source", "system")),
            action=record.data.get("action"),
            checkpoint=record.data.get("checkpoint"),
            run_id=record.data.get("run_id"),
            payload=record.data.get("payload"),
        )


def _event_record_id(session_id: str, event_id: str) -> str:
    return f"timeline.event::{session_id}::{event_id}"


class TimelineManager:
    """Persists timeline events per session/profile using the session store."""

    def __init__(
        self,
        *,
        session_store: Optional[SessionStore] = None,
        max_cached: int = 200,
    ) -> None:
        self._store = session_store or SessionStore()
        self._max_cached = max_cached
        self._session_id: Optional[str] = None
        self._profile: Optional[str] = None
        self._cache: List[OrchestratorEvent] = []

    def bind_session(
        self,
        session_id: Optional[str],
        *,
        profile: Optional[str] = None,
    ) -> None:
        self._session_id = session_id
        self._profile = profile
        self._cache = []
        if not session_id:
            return
        records = self._fetch_records(session_id, limit=self._max_cached)
        events: List[OrchestratorEvent] = []
        for record in records:
            entry = TimelineEntry.from_record(record)
            if entry is None:
                continue
            events.append(entry.to_event())
        events.sort(key=lambda event: event.timestamp)
        if len(events) > self._max_cached:
            events = events[-self._max_cached :]
        self._cache = events

    def append(self, event: OrchestratorEvent) -> None:
        if not self._session_id:
            return
        entry = TimelineEntry.from_event(
            event,
            session_id=self._session_id,
            profile=self._profile,
        )
        try:
            self._store.save(entry.to_record())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to persist timeline event: %s", exc)
        self._cache.append(event)
        if len(self._cache) > self._max_cached:
            self._cache = self._cache[-self._max_cached :]

    def recent(self, limit: int = 50) -> Iterable[OrchestratorEvent]:
        if not self._cache:
            return ()
        return self._cache[-limit:]

    def clear(self) -> None:
        if not self._session_id:
            return
        identifiers = [
            entry.identifier
            for entry in self._fetch_records(self._session_id, limit=500)
        ]
        for identifier in identifiers:
            try:
                self._store.delete(identifier)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to delete timeline event %s: %s", identifier, exc)
        self._cache = []

    def _fetch_records(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> List[SessionRecord]:
        try:
            records = self._store.list(
                profile=self._profile,
                category="timeline",
                limit=limit,
                include_expired=False,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load timeline records: %s", exc)
            return []
        prefix = f"timeline.event::{session_id}::"
        return [record for record in records if record.identifier.startswith(prefix)]
