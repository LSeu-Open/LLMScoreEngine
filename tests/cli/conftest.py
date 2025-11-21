"""Shared fixtures for CLI interaction tests."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pytest
from typer.testing import CliRunner

from llmscore.actions.base import ActionExecutionResult, ActionMetadata
from llmscore.workflows.models import WorkflowDefinition, WorkflowStep


class _StubRegistry:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []
        self.failures: Dict[str, Exception] = {}
        self.outputs: Dict[str, Dict[str, Any]] = {}
        self._available_actions: set[str] = set()

    def set_failure(self, action: str, exc: Exception) -> None:
        self.failures[action] = exc

    def set_output(self, action: str, payload: Dict[str, Any]) -> None:
        self.outputs[action] = payload

    def set_available_actions(self, names: Iterable[str]) -> None:
        self._available_actions = set(names)

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._available_actions))

    def run(self, action: str, *, inputs: Dict[str, Any] | None = None):
        payload = {
            "action": action,
            "inputs": dict(inputs or {}),
        }
        self.calls.append(payload)
        if action in self.failures:
            raise self.failures[action]
        output = self.outputs.get(action, {"status": "ok", **payload})
        metadata = ActionMetadata(
            name=action,
            title=f"Stub {action}",
            description="Stub action for CLI tests",
            domain="cli",
        )
        return ActionExecutionResult(metadata=metadata, output=output, duration=0.0)


class _StubWorkflowRegistry:
    def __init__(self, definition: Optional[WorkflowDefinition]):
        self.definition = definition
        self.requested: list[str] = []

    def set_definition(self, definition: Optional[WorkflowDefinition]) -> None:
        self.definition = definition

    def get(self, name: str) -> Optional[WorkflowDefinition]:
        self.requested.append(name)
        if self.definition and self.definition.name == name:
            return self.definition
        return None


@pytest.fixture(scope="session")
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def stub_registry(monkeypatch: pytest.MonkeyPatch) -> _StubRegistry:
    registry = _StubRegistry()
    monkeypatch.setattr("llmscore.__main__._build_registry", lambda: registry)
    return registry


@pytest.fixture
def sample_workflow_definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="nightly_score",
        description="Nightly scoring workflow",
        steps=
        [
            WorkflowStep(action="score.model", inputs={"model": "alpha"}),
            WorkflowStep(action="results.leaderboard", inputs={"limit": 3}),
        ],
    )


@pytest.fixture
def stub_workflow_registry(
    monkeypatch: pytest.MonkeyPatch,
    sample_workflow_definition: WorkflowDefinition,
) -> _StubWorkflowRegistry:
    registry = _StubWorkflowRegistry(sample_workflow_definition)
    monkeypatch.setattr("llmscore.__main__.WorkflowRegistry", lambda *_, **__: registry)
    monkeypatch.setattr("llmscore.__main__.SessionStore", lambda *_, **__: object())
    return registry
