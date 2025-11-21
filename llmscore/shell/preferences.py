"""Shell preference persistence helpers backed by the session store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from ..state.policies import SessionPolicyRegistry
from ..state.store import SessionRecord, SessionStore
from ..utils.logging import get_logger

LOGGER = get_logger("llmscore.shell.preferences")


@dataclass(slots=True)
class ShellPreferences:
    """Container for persisted shell preferences."""

    retention_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    updated_at: Optional[str] = None


def _config_record_id(profile: Optional[str]) -> str:
    suffix = profile or "default"
    return f"config::retention::{suffix}"


def load_preferences(store: SessionStore, profile: Optional[str]) -> ShellPreferences:
    """Load shell preferences stored for a given profile."""

    identifier = _config_record_id(profile)
    record = store.load(identifier)
    if not record or not isinstance(record.data, dict):
        return ShellPreferences()
    overrides = record.data.get("retention_overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}
    updated = record.data.get("updated")
    return ShellPreferences(
        retention_overrides={
            key: value
            for key, value in overrides.items()
            if isinstance(value, dict)
        },
        updated_at=updated if isinstance(updated, str) else None,
    )


def save_preferences(
    store: SessionStore,
    profile: Optional[str],
    preferences: ShellPreferences,
) -> ShellPreferences:
    """Persist shell preferences back into the session store."""

    preferences.updated_at = datetime.now(UTC).isoformat()
    record = SessionRecord(
        identifier=_config_record_id(profile),
        profile=profile,
        category="config",
        data={
            "retention_overrides": preferences.retention_overrides,
            "updated": preferences.updated_at,
        },
    )
    try:
        store.save(record)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to persist shell preferences: %s", exc)
    return preferences


def apply_preferences(
    store: SessionStore,
    preferences: ShellPreferences,
) -> SessionPolicyRegistry:
    """Apply preference-driven overrides to the session store policies."""

    active = store.get_policy_registry()
    registry = SessionPolicyRegistry(active.to_mapping())
    if preferences.retention_overrides:
        overrides = SessionPolicyRegistry.from_config(
            preferences.retention_overrides
        )
        for category, policy in overrides.to_mapping().items():
            registry.register(policy)
    store.set_policy_registry(registry)
    return registry
