"""Contextual suggestion engine for the interactive shell."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Deque, Dict, List, Optional

from ..state.store import SessionRecord, SessionStore
from ..utils.logging import get_logger

LOGGER = get_logger("llmscore.shell.suggestions")


@dataclass(slots=True)
class Suggestion:
    """Represents a contextual suggestion shown to the operator."""

    command: str
    title: str
    reason: str
    score: float = 0.0
    metadata: Optional[Dict[str, object]] = None


class SuggestionEngine:
    """Generates contextual hints based on session activity and telemetry."""

    def __init__(
        self,
        *,
        session_store: Optional[SessionStore] = None,
        recent_limit: int = 25,
    ) -> None:
        self._store = session_store or SessionStore()
        self._base_recent_limit = max(1, recent_limit)
        self._recent_actions: Deque[str] = deque(maxlen=self._base_recent_limit)
        self._usage = Counter[str]()
        self._ephemeral: List[Suggestion] = []
        self._pinned: List[Suggestion] = []
        self._session_id: Optional[str] = None
        self._profile: Optional[str] = None
        self._persistence_enabled = True
        self._policy_max_entries: Optional[int] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def bind_session(
        self,
        session_id: Optional[str],
        *,
        profile: Optional[str] = None,
    ) -> None:
        """Load state for the active session."""

        self._session_id = session_id
        self._profile = profile
        self._recent_actions = deque(
            maxlen=self._effective_recent_limit()
        )
        self._usage.clear()
        self._pinned = []
        self._refresh_policy_state()
        if not session_id:
            return

        if self._persistence_enabled:
            record = self._store.load(self._suggestion_record_id(session_id))
            if record:
                usage = record.data.get("usage", {})
                recent = record.data.get("recent", [])
                if isinstance(usage, dict):
                    self._usage.update({k: int(v) for k, v in usage.items()})
                if isinstance(recent, list):
                    for action in recent:
                        if isinstance(action, str):
                            self._recent_actions.append(action)
        self._load_pinned_suggestions()

    # ------------------------------------------------------------------
    # Recording events
    # ------------------------------------------------------------------
    def record_action(self, action: str) -> None:
        if not action:
            return
        self._usage[action] += 1
        self._recent_actions.appendleft(action)
        self._persist()

    def add_hint(
        self,
        command: str,
        *,
        title: str,
        reason: str,
        score: float = 0.0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        self._ephemeral.append(
            Suggestion(
                command=command,
                title=title,
                reason=reason,
                score=score,
                metadata=metadata,
            )
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get_suggestions(self, *, limit: int = 5) -> List[Suggestion]:
        candidates: List[Suggestion] = []
        candidates.extend(self._ephemeral)
        self._ephemeral.clear()

        for entry in self._pinned:
            candidates.append(entry)

        for action in list(self._recent_actions)[:limit]:
            candidates.append(
                Suggestion(
                    command=f"run {action}",
                    title=f"Re-run {action}",
                    reason="Recently executed",
                    score=0.6,
                )
            )

        for action, count in self._usage.most_common(limit):
            score = 0.5 + min(count / 10.0, 0.4)
            candidates.append(
                Suggestion(
                    command=f"run {action}",
                    title=f"Run {action}",
                    reason="Frequently used",
                    score=score,
                    metadata={"count": count},
                )
            )

        unique: Dict[str, Suggestion] = {}
        for suggestion in sorted(
            candidates,
            key=lambda item: item.score,
            reverse=True,
        ):
            unique.setdefault(suggestion.command, suggestion)
            if len(unique) >= limit:
                break
        return list(unique.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _suggestion_record_id(self, session_id: str) -> str:
        return f"ui.suggestions::{session_id}"

    def _persist(self) -> None:
        if not self._session_id or not self._persistence_enabled:
            return
        record = SessionRecord(
            identifier=self._suggestion_record_id(self._session_id),
            data={
                "usage": dict(self._usage),
                "recent": list(self._recent_actions),
                "updated": datetime.now(UTC).isoformat(),
            },
            profile=self._profile,
            category="suggestions",
        )
        try:
            self._store.save(record)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to persist suggestion state: %s", exc)

    def _load_pinned_suggestions(self) -> None:
        if not self._profile:
            return
        try:
            records = self._store.list(
                profile=self._profile,
                category="pinned_context",
                limit=50,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load session records: %s", exc)
            return
        pinned: List[Suggestion] = []
        for record in records:
            model = (
                record.data.get("model")
                if isinstance(record.data, dict)
                else None
            )
            if not isinstance(model, str):
                continue
            pinned.append(
                Suggestion(
                    command=f"results.show model={model}",
                    title=f"View {model} results",
                    reason="Pinned result",
                    score=0.9,
                    metadata={"pinned_at": record.data.get("pinned_at")},
                )
            )
        self._pinned = pinned

    def _refresh_policy_state(self) -> None:
        try:
            registry = self._store.get_policy_registry()
        except AttributeError:
            self._persistence_enabled = True
            self._policy_max_entries = None
            return
        policy = registry.resolve("suggestions")
        self._policy_max_entries = policy.max_entries
        self._persistence_enabled = (policy.max_entries or 0) != 0
        effective_limit = self._effective_recent_limit()
        if self._recent_actions.maxlen != effective_limit:
            snapshot = list(self._recent_actions)[:effective_limit]
            self._recent_actions = deque(snapshot, maxlen=effective_limit)

    def _effective_recent_limit(self) -> int:
        if self._policy_max_entries and self._policy_max_entries > 0:
            return max(1, min(self._base_recent_limit, self._policy_max_entries))
        return self._base_recent_limit


__all__ = ["Suggestion", "SuggestionEngine"]
