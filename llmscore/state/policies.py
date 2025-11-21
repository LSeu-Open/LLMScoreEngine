"""Retention and privacy policies applied to persisted session state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re
from typing import Dict, Iterable, Mapping, MutableMapping

__all__ = [
    "RetentionPolicy",
    "SessionPolicyRegistry",
    "DEFAULT_RETENTION_POLICIES",
    "parse_ttl_value",
]

_DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[smhdw])$")


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Defines retention behaviour for a session record category."""

    category: str
    ttl: timedelta | None = None
    max_entries: int | None = None
    protect_new_records: bool = False

    def expiry_for(self, reference: datetime | None = None) -> datetime | None:
        """Return an expiry timestamp based on the policy TTL."""

        if self.ttl is None:
            return None
        baseline = reference or datetime.now(UTC)
        return baseline + self.ttl


class SessionPolicyRegistry:
    """Registry of retention policies keyed by record category."""

    def __init__(self, policies: Mapping[str, RetentionPolicy] | None = None) -> None:
        self._policies: Dict[str, RetentionPolicy] = dict(DEFAULT_RETENTION_POLICIES)
        if policies:
            self._policies.update(policies)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Mapping[str, object]] | None
    ) -> "SessionPolicyRegistry":
        """Create a registry from a raw configuration mapping."""

        if not config:
            return cls()

        parsed: Dict[str, RetentionPolicy] = {}
        for category, payload in config.items():
            ttl = payload.get("ttl") if isinstance(payload, Mapping) else None  # type: ignore[assignment]
            max_entries = payload.get("max_entries") if isinstance(payload, Mapping) else None  # type: ignore[assignment]
            protect = payload.get("protect_new_records") if isinstance(payload, Mapping) else None
            parsed[category] = RetentionPolicy(
                category=category,
                ttl=parse_ttl_value(ttl),
                max_entries=int(max_entries) if max_entries is not None else None,
                protect_new_records=bool(protect) if protect is not None else False,
            )
        return cls(parsed)

    def categories(self) -> Iterable[str]:
        return self._policies.keys()

    def resolve(self, category: str | None) -> RetentionPolicy:
        key = category or "general"
        if key in self._policies:
            return self._policies[key]
        return self._policies["general"]

    def to_mapping(self) -> Mapping[str, RetentionPolicy]:
        return dict(self._policies)

    def register(self, policy: RetentionPolicy) -> None:
        self._policies[policy.category] = policy


def parse_ttl_value(raw: object) -> timedelta | None:
    """Parse a TTL value from configuration formats.

    Supported formats:
      * ``None`` or omitted => ``None`` (no expiry)
      * Integer/float => seconds
      * String with suffix (s, m, h, d, w)
    """

    if raw is None:
        return None
    if isinstance(raw, timedelta):
        return raw
    if isinstance(raw, (int, float)):
        return timedelta(seconds=float(raw))
    if isinstance(raw, str):
        match = _DURATION_RE.match(raw.strip().lower())
        if not match:
            raise ValueError(
                "Unsupported TTL format '%s'. Expected formats like '30d', '12h', or seconds as integer." % raw
            )
        value = int(match.group("value"))
        unit = match.group("unit")
        multiplier = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 604800,
        }[unit]
        return timedelta(seconds=value * multiplier)
    raise TypeError(
        f"Cannot parse TTL value from object of type {type(raw)!r}: {raw!r}"
    )


DEFAULT_RETENTION_POLICIES: MutableMapping[str, RetentionPolicy] = {
    "general": RetentionPolicy(category="general", ttl=None, max_entries=None),
    "timeline": RetentionPolicy(
        category="timeline",
        ttl=timedelta(days=14),
        max_entries=1000,
    ),
    "layout": RetentionPolicy(
        category="layout",
        ttl=None,
        max_entries=50,
    ),
    "pinned_context": RetentionPolicy(
        category="pinned_context",
        ttl=None,
        max_entries=200,
        protect_new_records=True,
    ),
    "export": RetentionPolicy(
        category="export",
        ttl=timedelta(days=30),
        max_entries=200,
    ),
    "suggestions": RetentionPolicy(
        category="suggestions",
        ttl=timedelta(days=7),
        max_entries=200,
    ),
    "scratchpad": RetentionPolicy(
        category="scratchpad",
        ttl=None,
        max_entries=500,
    ),
    "config": RetentionPolicy(
        category="config",
        ttl=None,
        max_entries=50,
    ),
}
