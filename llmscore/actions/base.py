"""Base classes and typing primitives for llmscore actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol


@dataclass(slots=True)
class ActionMetadata:
    """Describes an action exposed through the assistant."""

    name: str
    title: str
    description: str
    domain: str
    version: str = "0"
    deprecated: bool = False
    tags: tuple[str, ...] = ()


class ActionCallable(Protocol):
    """Protocol for callable action implementations."""

    def __call__(self, **inputs: Any) -> Any:
        ...  # pragma: no cover - structural type


ActionBeforeHook = Callable[["ActionDefinition", Dict[str, Any]], None]
ActionAfterHook = Callable[["ActionDefinition", Dict[str, Any], Any], None]
UndoCallable = Callable[["ActionDefinition", Dict[str, Any], Any], None]


class ActionExecutionError(RuntimeError):
    """Raised when an action handler fails during execution."""

    def __init__(self, metadata: ActionMetadata, original: Exception) -> None:
        message = f"Action '{metadata.name}' failed: {original}"
        super().__init__(message)
        self.metadata = metadata
        self.original = original


@dataclass(slots=True)
class ActionExecutionResult:
    """Captures the outcome of invoking an action handler."""

    metadata: ActionMetadata
    output: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class ActionDefinition:
    """Container for action metadata, schema, and handler."""

    metadata: ActionMetadata
    handler: ActionCallable
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    examples: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    on_complete: Optional[Callable[[Any], None]] = None
    before_hooks: tuple[ActionBeforeHook, ...] = ()
    after_hooks: tuple[ActionAfterHook, ...] = ()
    undo: Optional[UndoCallable] = None

    @property
    def supports_undo(self) -> bool:
        return self.undo is not None


__all__ = [
    "ActionMetadata",
    "ActionCallable",
    "ActionDefinition",
    "ActionBeforeHook",
    "ActionAfterHook",
    "UndoCallable",
    "ActionExecutionError",
    "ActionExecutionResult",
]
