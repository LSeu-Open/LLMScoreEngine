"""Scratchpad utilities for quick notes and artifact references."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from ..state.store import SessionRecord, SessionStore
from ..utils.logging import get_logger

LOGGER = get_logger("llmscore.shell.scratchpad")


@dataclass(slots=True)
class ScratchpadNote:
    """A persisted scratchpad entry."""

    identifier: str
    content: str
    created_at: str
    tags: tuple[str, ...] = ()


class ScratchpadManager:
    """Provides note-taking facilities backed by the session store."""

    def __init__(
        self,
        *,
        session_store: Optional[SessionStore] = None,
    ) -> None:
        self._store = session_store or SessionStore()
        self._session_id: Optional[str] = None
        self._profile: Optional[str] = None
        self._notes: List[ScratchpadNote] = []

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def bind_session(
        self,
        session_id: Optional[str],
        *,
        profile: Optional[str] = None,
    ) -> None:
        self._session_id = session_id
        self._profile = profile
        self._notes = []
        if not session_id:
            return
        record = self._store.load(self._scratchpad_record_id(session_id))
        if not record:
            return
        notes = (
            record.data.get("notes", [])
            if isinstance(record.data, dict)
            else []
        )
        for entry in notes:
            identifier = entry.get("id")
            content = entry.get("content")
            created_at = entry.get("created_at")
            tags = tuple(entry.get("tags", ()))
            if all(
                isinstance(value, str)
                for value in (identifier, content, created_at)
            ):
                self._notes.append(
                    ScratchpadNote(
                        identifier=identifier,
                        content=content,
                        created_at=created_at,
                        tags=tags,
                    )
                )

    # ------------------------------------------------------------------
    # Note operations
    # ------------------------------------------------------------------
    def add(self, content: str, *, tags: Sequence[str] = ()) -> ScratchpadNote:
        note = ScratchpadNote(
            identifier=str(uuid4()),
            content=content,
            created_at=datetime.now(UTC).isoformat(),
            tags=tuple(tags),
        )
        self._notes.insert(0, note)
        self._persist()
        return note

    def list(self) -> Iterable[ScratchpadNote]:
        return tuple(self._notes)

    def remove(self, identifier: str) -> bool:
        for index, note in enumerate(self._notes):
            if note.identifier == identifier:
                self._notes.pop(index)
                self._persist()
                return True
        return False

    def clear(self) -> None:
        self._notes.clear()
        self._persist()

    def export(self) -> str:
        lines: List[str] = []
        for note in self._notes:
            tag_text = " ".join(note.tags) if note.tags else ""
            header = f"[{note.created_at}] {tag_text}".strip()
            lines.append(header)
            lines.append(note.content)
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _scratchpad_record_id(self, session_id: str) -> str:
        return f"scratchpad::{session_id}"

    def _persist(self) -> None:
        if not self._session_id:
            return
        payload = {
            "notes": [self._note_to_dict(note) for note in self._notes],
            "updated": datetime.now(UTC).isoformat(),
        }
        record = SessionRecord(
            identifier=self._scratchpad_record_id(self._session_id),
            data=payload,
            profile=self._profile,
        )
        try:
            self._store.save(record)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to persist scratchpad: %s", exc)

    @staticmethod
    def _note_to_dict(note: ScratchpadNote) -> Dict[str, object]:
        return {
            "id": note.identifier,
            "content": note.content,
            "created_at": note.created_at,
            "tags": list(note.tags),
        }


__all__ = ["ScratchpadManager", "ScratchpadNote"]
