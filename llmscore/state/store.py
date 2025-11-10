"""Session store abstraction for persisting CLI state."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import Column, JSON


SCHEMA_VERSION = 1


@dataclass
class SessionRecord:
    """Public representation of a session record."""

    identifier: str
    data: Dict[str, Any]
    profile: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SessionRecordModel(SQLModel, table=True):
    """SQLModel table mapping for the session store."""

    id: str = Field(primary_key=True)
    profile: Optional[str] = Field(default=None, index=True)
    data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False, default=dict),
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SessionStore:
    """SQLite-backed session store with lightweight migrations."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        schema_version: int = SCHEMA_VERSION,
    ) -> None:
        self.path = path or Path("~/.llmscore/session.db").expanduser()
        self.schema_version = schema_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.path}", echo=False)
        self._initialize_database()

    # ------------------------------------------------------------------
    # Initialization & migrations
    # ------------------------------------------------------------------
    def _initialize_database(self) -> None:
        with self.engine.connect() as connection:
            cursor = connection.exec_driver_sql("PRAGMA user_version")
            current = cursor.scalar() or 0
            if current == 0:
                SQLModel.metadata.create_all(self.engine)
                connection.exec_driver_sql(
                    f"PRAGMA user_version = {self.schema_version}"
                )
            elif current > self.schema_version:
                raise RuntimeError(
                    "Session database schema version is newer than supported."
                )
            elif current < self.schema_version:
                self._apply_migrations(connection, current)

    def _apply_migrations(self, connection, current_version: int) -> None:
        """Apply in-place migrations to reach the configured schema version."""

        # No additional migrations yet; hook left in place for future updates.
        SQLModel.metadata.create_all(self.engine)
        connection.exec_driver_sql(
            f"PRAGMA user_version = {self.schema_version}"
        )

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def save(self, record: SessionRecord) -> SessionRecord:
        now = datetime.utcnow()
        with Session(self.engine) as session:
            model = session.get(SessionRecordModel, record.identifier)
            if model is None:
                model = SessionRecordModel(
                    id=record.identifier,
                    profile=record.profile,
                    data=record.data,
                    created_at=now,
                    updated_at=now,
                )
                session.add(model)
            else:
                model.data = record.data
                model.profile = record.profile
                model.updated_at = now
            session.commit()
            session.refresh(model)
        return self._to_record(model)

    def load(self, identifier: str) -> Optional[SessionRecord]:
        with Session(self.engine) as session:
            model = session.get(SessionRecordModel, identifier)
            if not model:
                return None
            return self._to_record(model)

    def list(
        self,
        *,
        profile: Optional[str] = None,
        limit: int | None = None,
    ) -> List[SessionRecord]:
        statement = select(SessionRecordModel).order_by(
            SessionRecordModel.updated_at.desc()
        )
        if profile:
            statement = statement.where(SessionRecordModel.profile == profile)
        if limit:
            statement = statement.limit(limit)
        with Session(self.engine) as session:
            models = session.exec(statement).all()
            return [self._to_record(model) for model in models]

    def delete(self, identifier: str) -> None:
        with Session(self.engine) as session:
            model = session.get(SessionRecordModel, identifier)
            if model:
                session.delete(model)
                session.commit()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def backup(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path, destination)
        return destination

    def close(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _to_record(model: SessionRecordModel) -> SessionRecord:
        return SessionRecord(
            identifier=model.id,
            data=model.data,
            profile=model.profile,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


__all__ = [
    "SessionRecord",
    "SessionStore",
    "SessionRecordModel",
    "SCHEMA_VERSION",
]
