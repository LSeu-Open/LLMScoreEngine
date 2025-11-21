"""Event primitives used by the orchestration layer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, DefaultDict, List, Literal, Mapping, Optional
from uuid import uuid4


EventKind = Literal[
    "info",
    "warning",
    "error",
    "progress",
]

EventSource = Literal[
    "system",
    "action",
    "controller",
    "workflow",
]


@dataclass(slots=True)
class OrchestratorEvent:
    """Represents an event emitted during task execution."""

    kind: EventKind
    message: str
    timestamp: datetime
    payload: Optional[Mapping[str, Any]] = None
    source: EventSource = "system"
    action: Optional[str] = None
    checkpoint: Optional[str] = None
    run_id: Optional[str] = None
    id: str = field(default_factory=lambda: uuid4().hex)


EventSubscriber = Callable[[OrchestratorEvent], None]


class EventBus:
    """Simple publish/subscribe event bus for orchestration events."""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[
            Optional[EventKind],
            List[EventSubscriber],
        ] = defaultdict(list)

    def subscribe(
        self,
        subscriber: EventSubscriber,
        *,
        kind: Optional[EventKind] = None,
    ) -> None:
        self._subscribers[kind].append(subscriber)

    def unsubscribe(
        self,
        subscriber: EventSubscriber,
        *,
        kind: Optional[EventKind] = None,
    ) -> None:
        listeners = self._subscribers.get(kind)
        if not listeners:
            return
        if subscriber in listeners:
            listeners.remove(subscriber)

    def emit(self, event: OrchestratorEvent) -> None:
        for listener in list(self._subscribers.get(None, [])):
            listener(event)
        for listener in list(self._subscribers.get(event.kind, [])):
            listener(event)

    def clear(self) -> None:
        self._subscribers.clear()


__all__ = ["EventKind", "OrchestratorEvent", "EventSubscriber", "EventBus"]
