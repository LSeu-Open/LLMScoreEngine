"""Task orchestration controller for the interactive shell."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Deque, List, Optional

from .events import EventBus, EventKind, OrchestratorEvent


@dataclass(slots=True)
class TaskCheckpoint:
    """Represents a step in the orchestration timeline."""

    identifier: str
    description: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class TaskController:
    """Maintains task sequencing, checkpoints, and emitted events."""

    event_bus: Optional[EventBus] = None
    checkpoints: Deque[TaskCheckpoint] = field(default_factory=deque)
    events: List[OrchestratorEvent] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Timeline management
    # ------------------------------------------------------------------
    def add_checkpoint(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        """Append a checkpoint to the timeline without changing status."""

        self.checkpoints.append(checkpoint)
        self._emit(
            "info",
            f"Checkpoint queued: {checkpoint.description}",
            {
                "id": checkpoint.identifier,
                "status": checkpoint.status,
            },
        )
        return checkpoint

    def start(self, identifier: str, description: str) -> TaskCheckpoint:
        checkpoint = TaskCheckpoint(
            identifier=identifier,
            description=description,
            status="running",
            started_at=datetime.now(UTC),
        )
        self.checkpoints.append(checkpoint)
        self._emit(
            "progress",
            f"Started: {description}",
            {
                "id": identifier,
                "status": checkpoint.status,
            },
        )
        return checkpoint

    def complete(
        self,
        identifier: str,
        output: Any | None = None,
    ) -> Optional[TaskCheckpoint]:
        checkpoint = self._locate(identifier)
        if not checkpoint:
            return None
        checkpoint.status = "completed"
        checkpoint.completed_at = datetime.now(UTC)
        checkpoint.error = None
        payload = {
            "id": identifier,
            "status": checkpoint.status,
        }
        if output is not None:
            payload["output"] = output
        self._emit("progress", f"Completed: {checkpoint.description}", payload)
        return checkpoint

    def fail(
        self,
        identifier: str,
        error: Exception,
    ) -> Optional[TaskCheckpoint]:
        checkpoint = self._locate(identifier)
        if not checkpoint:
            return None
        checkpoint.status = "failed"
        checkpoint.completed_at = datetime.now(UTC)
        checkpoint.error = str(error)
        self._emit(
            "error",
            f"Failed: {checkpoint.description}",
            {
                "id": identifier,
                "status": checkpoint.status,
                "error": checkpoint.error,
            },
        )
        return checkpoint

    def record_event(self, event: OrchestratorEvent) -> None:
        self.events.append(event)
        if self.event_bus:
            self.event_bus.emit(event)

    def current(self) -> Optional[TaskCheckpoint]:
        return self.checkpoints[-1] if self.checkpoints else None

    def clear(self) -> None:
        self.checkpoints.clear()
        self.events.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _locate(self, identifier: str) -> Optional[TaskCheckpoint]:
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.identifier == identifier:
                return checkpoint
        return None

    def _emit(
        self,
        kind: EventKind,
        message: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        event = OrchestratorEvent(
            kind=kind,
            message=message,
            timestamp=datetime.now(UTC),
            payload=payload,
        )
        self.events.append(event)
        if self.event_bus:
            self.event_bus.emit(event)


__all__ = ["TaskCheckpoint", "TaskController"]
