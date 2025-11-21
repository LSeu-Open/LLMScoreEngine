"""Session store abstraction for persisting CLI state."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select, delete
from sqlalchemy import Column, JSON
from sqlalchemy.exc import OperationalError

from .policies import (
    DEFAULT_RETENTION_POLICIES,
    RetentionPolicy,
    SessionPolicyRegistry,
)


SCHEMA_VERSION = 2


_CATEGORY_PREFIXES: Mapping[str, str] = {
    "timeline::": "timeline",
    "timeline.event::": "timeline",
    "layout::": "layout",
    "ui.layout::": "layout",
    "context.pin::": "pinned_context",
    "results.pin::": "pinned_context",
    "export::": "export",
    "ui.suggestions::": "suggestions",
    "scratchpad::": "scratchpad",
    "config::": "config",
}


@dataclass
class SessionRecord:
    """Public representation of a session record."""

    identifier: str
    data: Dict[str, Any]
    profile: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class SessionRecordModel(SQLModel, table=True):
    """SQLModel table mapping for the session store."""

    id: str = Field(primary_key=True)
    profile: Optional[str] = Field(default=None, index=True)
    category: str = Field(default="general", index=True)
    data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False, default=dict),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = Field(default=None, nullable=True)


class SessionStore:
    """SQLite-backed session store with lightweight migrations."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        schema_version: int = SCHEMA_VERSION,
        policies: Optional[SessionPolicyRegistry] = None,
    ) -> None:
        self.path = path or Path("~/.llmscore/session.db").expanduser()
        self.schema_version = schema_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.path}", echo=False)
        self._policies = policies or SessionPolicyRegistry(DEFAULT_RETENTION_POLICIES)
        self._initialize_database()

    # ------------------------------------------------------------------
    # Policy registry management
    # ------------------------------------------------------------------
    def get_policy_registry(self) -> SessionPolicyRegistry:
        """Return the currently active retention policy registry."""

        return self._policies

    def set_policy_registry(self, registry: SessionPolicyRegistry) -> None:
        """Replace the active retention policy registry."""

        self._policies = registry

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

        current = current_version
        if current < 1:
            SQLModel.metadata.create_all(self.engine)
            current = 1
            connection.exec_driver_sql("PRAGMA user_version = 1")
        if current < 2:
            self._migrate_to_v2(connection)
            current = 2
        if current != self.schema_version:
            connection.exec_driver_sql(
                f"PRAGMA user_version = {self.schema_version}"
            )

    def _migrate_to_v2(self, connection) -> None:
        """Add category & expiry tracking for retention policies."""

        try:
            connection.exec_driver_sql(
                "ALTER TABLE sessionrecordmodel ADD COLUMN category TEXT DEFAULT 'general'"
            )
        except OperationalError:
            pass
        try:
            connection.exec_driver_sql(
                "ALTER TABLE sessionrecordmodel ADD COLUMN expires_at DATETIME"
            )
        except OperationalError:
            pass
        connection.exec_driver_sql(
            """
            UPDATE sessionrecordmodel
            SET category = CASE
                WHEN id LIKE 'timeline::%' THEN 'timeline'
                WHEN id LIKE 'timeline.event::%' THEN 'timeline'
                WHEN id LIKE 'layout::%' THEN 'layout'
                WHEN id LIKE 'ui.layout::%' THEN 'layout'
                WHEN id LIKE 'context.pin::%' THEN 'pinned_context'
                WHEN id LIKE 'results.pin::%' THEN 'pinned_context'
                WHEN id LIKE 'export::%' THEN 'export'
                ELSE 'general'
            END
            """
        )

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def save(self, record: SessionRecord) -> SessionRecord:
        now = datetime.now(UTC)
        category = self._resolve_category(record)
        policy = self._policies.resolve(category)
        expiry = policy.expiry_for(now)
        with Session(self.engine) as session:
            model = session.get(SessionRecordModel, record.identifier)
            if model is None:
                model = SessionRecordModel(
                    id=record.identifier,
                    profile=record.profile,
                    data=record.data,
                    created_at=now,
                    updated_at=now,
                    category=category,
                    expires_at=expiry,
                )
                session.add(model)
            else:
                model.data = record.data
                model.profile = record.profile
                model.updated_at = now
                model.category = category
                model.expires_at = expiry
            session.commit()
            session.refresh(model)
        self._enforce_max_entries(policy, category, record.profile)
        return self._to_record(model)

    def load(self, identifier: str) -> Optional[SessionRecord]:
        with Session(self.engine) as session:
            model = session.get(SessionRecordModel, identifier)
            if not model:
                return None
            if model.expires_at and _as_utc(model.expires_at) <= datetime.now(UTC):
                return None
            return self._to_record(model)

    def list(
        self,
        *,
        profile: Optional[str] = None,
        category: Optional[str] = None,
        limit: int | None = None,
        include_expired: bool = False,
    ) -> List[SessionRecord]:
        statement = select(SessionRecordModel).order_by(
            SessionRecordModel.updated_at.desc()
        )
        if profile:
            statement = statement.where(SessionRecordModel.profile == profile)
        if category:
            statement = statement.where(SessionRecordModel.category == category)
        if not include_expired:
            statement = statement.where(
                (SessionRecordModel.expires_at.is_(None))
                | (
                    SessionRecordModel.expires_at
                    > datetime.now(UTC).replace(tzinfo=None)
                )
            )
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
    def purge_expired(
        self, *, reference_time: Optional[datetime] = None
    ) -> int:
        """Remove all expired records and return the count removed."""

        reference = reference_time or datetime.now(UTC)
        reference_naive = reference.replace(tzinfo=None)
        with Session(self.engine) as session:
            result = session.exec(
                delete(SessionRecordModel).where(
                    SessionRecordModel.expires_at.is_not(None),
                    SessionRecordModel.expires_at <= reference_naive,
                )
            )
            session.commit()
            return int(result.rowcount or 0)

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
            category=model.category,
            created_at=model.created_at,
            updated_at=model.updated_at,
            expires_at=model.expires_at,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_category(self, record: SessionRecord) -> str:
        if record.category:
            return record.category
        return self._infer_category(record.identifier)

    @staticmethod
    def _infer_category(identifier: str) -> str:
        for prefix, category in _CATEGORY_PREFIXES.items():
            if identifier.startswith(prefix):
                return category
        return "general"

    def _enforce_max_entries(
        self,
        policy: RetentionPolicy,
        category: str,
        profile: Optional[str],
    ) -> int:
        if not policy.max_entries:
            return 0
        with Session(self.engine) as session:
            statement = select(SessionRecordModel.id).where(
                SessionRecordModel.category == category
            )
            if profile:
                statement = statement.where(SessionRecordModel.profile == profile)
            stale_ids = session.exec(
                statement.order_by(SessionRecordModel.updated_at.desc()).offset(
                    policy.max_entries
                )
            ).all()
            if not stale_ids:
                return 0
            session.exec(
                delete(SessionRecordModel).where(
                    SessionRecordModel.id.in_(stale_ids)
                )
            )
            session.commit()
            return len(stale_ids)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "SessionRecord",
    "SessionStore",
    "SessionRecordModel",
    "SCHEMA_VERSION",
]
