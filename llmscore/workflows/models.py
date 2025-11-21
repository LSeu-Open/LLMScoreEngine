"""Workflow metadata models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(slots=True)
class WorkflowStep:
    """Single action invocation within a workflow."""

    action: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "inputs": self.inputs,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        return cls(
            action=data["action"],
            inputs=dict(data.get("inputs", {})),
            description=data.get("description"),
        )


@dataclass(slots=True)
class WorkflowDefinition:
    """Represents a persisted workflow template."""

    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    version: str = "1.0"
    tags: Tuple[str, ...] = ()
    author: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    source: str | None = None

    def validate(self) -> None:
        if not self.steps:
            raise ValueError("Workflow must contain at least one step")
        for step in self.steps:
            if not step.action:
                raise ValueError("Workflow steps must include an action name")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": list(self.tags),
            "author": self.author,
            "created_at": self.created_at,
            "source": self.source,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        raw_steps = data.get("steps", [])
        steps = [WorkflowStep.from_dict(item) for item in raw_steps]
        definition = cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            version=data.get("version", "1.0"),
            tags=tuple(data.get("tags", ())),
            author=data.get("author"),
            created_at=data.get(
                "created_at",
                datetime.now(UTC).isoformat(),
            ),
            source=data.get("source"),
        )
        definition.validate()
        return definition

    def required_actions(self) -> Sequence[str]:
        return [step.action for step in self.steps]

    def with_steps(
        self,
        steps: Iterable[WorkflowStep],
    ) -> "WorkflowDefinition":
        updated = WorkflowDefinition(
            name=self.name,
            description=self.description,
            steps=list(steps),
            version=self.version,
            tags=self.tags,
            author=self.author,
            created_at=self.created_at,
            source=self.source,
        )
        updated.validate()
        return updated


__all__ = [
    "WorkflowDefinition",
    "WorkflowStep",
]
