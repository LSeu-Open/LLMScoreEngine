from datetime import datetime
from pathlib import Path
from typing import Dict

from llmscore.state.store import SessionRecord
from llmscore.workflows.models import WorkflowDefinition, WorkflowStep
from llmscore.workflows.registry import WorkflowRegistry


class InMemorySessionStore:
    def __init__(self) -> None:
        self._records: Dict[str, SessionRecord] = {}

    def save(self, record: SessionRecord) -> SessionRecord:
        now = datetime.utcnow()
        stored = SessionRecord(
            identifier=record.identifier,
            data=record.data,
            profile=record.profile,
            created_at=record.created_at or now,
            updated_at=now,
        )
        self._records[record.identifier] = stored
        return stored

    def load(self, identifier: str) -> SessionRecord | None:
        return self._records.get(identifier)

    def list(
        self,
        *,
        profile: str | None = None,
        limit: int | None = None,
    ) -> list[SessionRecord]:
        records = list(self._records.values())
        if profile is not None:
            records = [
                record
                for record in records
                if record.profile == profile
            ]
        records.sort(
            key=lambda record: record.updated_at or datetime.min,
            reverse=True,
        )
        if limit is not None:
            records = records[:limit]
        return records

    def delete(self, identifier: str) -> None:
        self._records.pop(identifier, None)


def _sample_workflow(name: str = "example") -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        description="Sample workflow",
        steps=[
            WorkflowStep(
                action="score.batch",
                inputs={"models": ["demo"]},
            )
        ],
        tags=("test",),
        author="unit-test",
    )


def test_workflow_registry_save_and_list(tmp_path: Path) -> None:
    store = InMemorySessionStore()
    registry = WorkflowRegistry(store)

    definition = _sample_workflow()
    registry.save(definition, overwrite=True)

    workflows = registry.list()
    assert len(workflows) == 1
    saved = workflows[0]
    assert saved.name == definition.name
    assert saved.description == definition.description
    assert saved.steps[0].action == "score.batch"


def test_workflow_registry_export_import(tmp_path: Path) -> None:
    store = InMemorySessionStore()
    registry = WorkflowRegistry(store)

    definition = _sample_workflow("pipeline-a")
    registry.save(definition, overwrite=True)

    export_path = tmp_path / "workflow.json"
    registry.export_to_path("pipeline-a", export_path)
    assert export_path.exists()

    new_store = InMemorySessionStore()
    new_registry = WorkflowRegistry(new_store)

    imported = new_registry.import_from_path(export_path, source="export-test")
    fetched = new_registry.get("pipeline-a")

    assert imported.name == "pipeline-a"
    assert fetched is not None
    assert fetched.source == "export-test"
    assert fetched.steps[0].inputs["models"] == ["demo"]
