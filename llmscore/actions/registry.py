"""Action registry and execution utilities for llmscore."""

from __future__ import annotations

import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
)

from ..orchestrator.controller import TaskController
from ..orchestrator.events import EventBus, EventKind, OrchestratorEvent
from .base import (
    ActionDefinition,
    ActionExecutionError,
    ActionExecutionResult,
)


class ActionRegistry:
    """Stores registered actions and orchestrates their execution."""

    def __init__(self, *, event_bus: Optional[EventBus] = None) -> None:
        self._actions: Dict[str, ActionDefinition] = OrderedDict()
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        definition: ActionDefinition,
        *,
        replace: bool = False,
    ) -> None:
        """Register an action definition.

        Args:
            definition: The action definition to register.
            replace: Allow replacement when the action already exists.
        """

        name = definition.metadata.name
        if not replace and name in self._actions:
            raise ValueError(f"Action '{name}' already registered")
        self._actions[name] = definition

    def deregister(self, name: str) -> None:
        self._actions.pop(name, None)

    def get(self, name: str) -> ActionDefinition:
        return self._actions[name]

    def __contains__(self, name: str) -> bool:
        return name in self._actions

    def __iter__(self) -> Iterator[ActionDefinition]:
        return iter(self._actions.values())

    def names(self) -> Iterable[str]:
        return self._actions.keys()

    def by_domain(self, domain: str) -> Iterable[ActionDefinition]:
        return (
            definition
            for definition in self._actions.values()
            if definition.metadata.domain == domain
        )

    def bind_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        name: str,
        *,
        inputs: Optional[Mapping[str, Any]] = None,
        controller: Optional[TaskController] = None,
    ) -> ActionExecutionResult:
        definition = self.get(name)
        arguments: MutableMapping[str, Any] = dict(inputs or {})
        checkpoint = None
        if controller:
            checkpoint = controller.start(name, definition.metadata.title)
        self._emit(
            "progress",
            f"Dispatching action '{name}'",
            {"id": name, "title": definition.metadata.title},
        )

        start = time.perf_counter()
        try:
            for hook in definition.before_hooks:
                hook(definition, arguments)

            output = definition.handler(**arguments)

            if definition.on_complete:
                definition.on_complete(output)

            for hook in definition.after_hooks:
                hook(definition, arguments, output)

            duration = time.perf_counter() - start
            result = ActionExecutionResult(
                metadata=definition.metadata,
                output=output,
                duration=duration,
            )

            if controller and checkpoint:
                controller.complete(checkpoint.identifier, output=output)

            self._emit(
                "progress",
                f"Action '{name}' completed",
                {
                    "id": name,
                    "duration": duration,
                    "status": "completed",
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001 - propagate wrapped error
            duration = time.perf_counter() - start
            if controller and checkpoint:
                controller.fail(checkpoint.identifier, exc)
            self._emit(
                "error",
                f"Action '{name}' failed",
                {
                    "id": name,
                    "status": "failed",
                    "error": str(exc),
                },
            )
            result = ActionExecutionResult(
                metadata=definition.metadata,
                error=exc,
                duration=duration,
            )
            raise ActionExecutionError(definition.metadata, exc) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit(
        self,
        kind: EventKind,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._event_bus:
            return
        event = OrchestratorEvent(
            kind=kind,
            message=message,
            timestamp=datetime.now(UTC),
            payload=payload,
        )
        self._event_bus.emit(event)


__all__ = ["ActionRegistry"]
