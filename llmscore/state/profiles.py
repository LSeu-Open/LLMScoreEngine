"""Profile management for llmscore sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import List, Optional

from sqlmodel import Field, Session, SQLModel, select

from .store import SessionStore


@dataclass(slots=True)
class Profile:
    """Represents a user or workspace profile."""

    name: str
    workspace_path: str
    default_session: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProfileModel(SQLModel, table=True):
    """Database representation of a profile."""

    name: str = Field(primary_key=True)
    workspace_path: str
    default_session: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProfileManager:
    """SQLite-backed profile registry."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self.engine = store.engine
        SQLModel.metadata.create_all(self.engine)

    def add(self, profile: Profile) -> Profile:
        now = datetime.now(UTC)
        with Session(self.engine) as session:
            model = session.get(ProfileModel, profile.name)
            if model is None:
                model = ProfileModel(
                    name=profile.name,
                    workspace_path=profile.workspace_path,
                    default_session=profile.default_session,
                    created_at=now,
                    updated_at=now,
                )
                session.add(model)
            else:
                model.workspace_path = profile.workspace_path
                model.default_session = profile.default_session
                model.updated_at = now
            session.commit()
            session.refresh(model)
        return self._to_profile(model)

    def get(self, name: str) -> Optional[Profile]:
        with Session(self.engine) as session:
            model = session.get(ProfileModel, name)
            if not model:
                return None
            return self._to_profile(model)

    def list(self) -> List[Profile]:
        with Session(self.engine) as session:
            statement = select(ProfileModel).order_by(ProfileModel.name)
            models = session.exec(statement).all()
            return [self._to_profile(model) for model in models]

    def remove(self, name: str) -> None:
        with Session(self.engine) as session:
            model = session.get(ProfileModel, name)
            if model:
                session.delete(model)
                session.commit()

    def set_default_session(
        self,
        name: str,
        session_id: Optional[str],
    ) -> Optional[Profile]:
        with Session(self.engine) as session:
            model = session.get(ProfileModel, name)
            if not model:
                return None
            model.default_session = session_id
            model.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(model)
            return self._to_profile(model)

    @staticmethod
    def _to_profile(model: ProfileModel) -> Profile:
        return Profile(
            name=model.name,
            workspace_path=model.workspace_path,
            default_session=model.default_session,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


__all__ = ["Profile", "ProfileModel", "ProfileManager"]
