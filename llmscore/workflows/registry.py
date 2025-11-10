"""Workflow registry for saving, importing, and exporting definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from ..state.store import SessionRecord, SessionStore
from ..utils.logging import get_logger
from .models import WorkflowDefinition, WorkflowStep

LOGGER = get_logger("llmscore.workflows.registry")


class WorkflowRegistry:
    """Persistence layer for workflow definitions."""

    def __init__(
        self,
        store: SessionStore,
        *,
        profile: Optional[str] = None,
    ) -> None:
        self.store = store
        self.profile = profile

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def list(self) -> List[WorkflowDefinition]:
        records = self.store.list(profile=self.profile)
        definitions: List[WorkflowDefinition] = []
        for record in records:
            if not record.identifier.startswith("workflow::"):
                continue
            try:
                definitions.append(
                    WorkflowDefinition.from_dict(record.data)
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Skipping workflow %s due to error: %s",
                    record.identifier,
                    exc,
                )
        return definitions

    def get(self, name: str) -> Optional[WorkflowDefinition]:
        record = self.store.load(self._record_id(name))
        if not record:
            return None
        try:
            return WorkflowDefinition.from_dict(record.data)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load workflow %s: %s", name, exc)
            return None

    def save(
        self,
        definition: WorkflowDefinition,
        *,
        overwrite: bool = False,
    ) -> WorkflowDefinition:
        definition.validate()
        if not overwrite and self.get(definition.name):
            raise ValueError(f"Workflow '{definition.name}' already exists")
        record = SessionRecord(
            identifier=self._record_id(definition.name),
            data=definition.to_dict(),
            profile=self.profile,
        )
        self.store.save(record)
        LOGGER.info("Saved workflow '%s'", definition.name)
        return definition

    def delete(self, name: str) -> None:
        self.store.delete(self._record_id(name))
        LOGGER.info("Deleted workflow '%s'", name)

    # ------------------------------------------------------------------
    # Import / export helpers
    # ------------------------------------------------------------------
    def import_from_path(
        self,
        path: Path,
        *,
        overwrite: bool = False,
        source: Optional[str] = None,
    ) -> WorkflowDefinition:
        data = self._load_document(path)
        definition = WorkflowDefinition.from_dict(data)
        if source:
            definition.source = source
        self.save(definition, overwrite=overwrite)
        return definition

    def export_to_path(self, name: str, path: Path) -> Path:
        definition = self.get(name)
        if not definition:
            raise ValueError(f"Workflow '{name}' not found")
        payload = definition.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        LOGGER.info("Exported workflow '%s' to %s", name, path)
        return path

    def import_from_payload(
        self,
        payload: Dict[str, object],
        *,
        overwrite: bool = False,
        source: Optional[str] = None,
    ) -> WorkflowDefinition:
        definition = WorkflowDefinition.from_dict(payload)
        if source:
            definition.source = source
        return self.save(definition, overwrite=overwrite)

    def create_from_steps(
        self,
        name: str,
        steps: Iterable[WorkflowStep],
        *,
        description: str = "",
        tags: Optional[Iterable[str]] = None,
        overwrite: bool = False,
    ) -> WorkflowDefinition:
        definition = WorkflowDefinition(
            name=name,
            description=description,
            steps=list(steps),
            tags=tuple(tags or ()),
        )
        return self.save(definition, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_document(path: Path) -> Dict[str, object]:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(text)
        return json.loads(text)

    @staticmethod
    def _record_id(name: str) -> str:
        return f"workflow::{name}"


__all__ = ["WorkflowRegistry"]
