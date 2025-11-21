"""Tests for session retention policy configuration."""

from datetime import timedelta
import pytest

from llmscore.state.policies import (
    DEFAULT_RETENTION_POLICIES,
    RetentionPolicy,
    SessionPolicyRegistry,
    parse_ttl_value,
)


def test_parse_ttl_value_accepts_seconds_as_int() -> None:
    assert parse_ttl_value(60) == timedelta(seconds=60)


def test_parse_ttl_value_accepts_suffix_string() -> None:
    assert parse_ttl_value("2h") == timedelta(hours=2)


def test_parse_ttl_value_returns_none_for_none() -> None:
    assert parse_ttl_value(None) is None


def test_parse_ttl_value_rejects_invalid_format() -> None:
    with pytest.raises(ValueError):
        parse_ttl_value("10years")


def test_registry_defaults_are_present() -> None:
    registry = SessionPolicyRegistry()
    for key, policy in DEFAULT_RETENTION_POLICIES.items():
        resolved = registry.resolve(key)
        assert resolved.category == policy.category
        assert resolved.ttl == policy.ttl
        assert resolved.max_entries == policy.max_entries


def test_registry_from_config_overrides_defaults() -> None:
    registry = SessionPolicyRegistry.from_config(
        {
            "timeline": {
                "ttl": "1d",
                "max_entries": 10,
            }
        }
    )
    resolved = registry.resolve("timeline")
    assert resolved.ttl == timedelta(days=1)
    assert resolved.max_entries == 10


def test_registry_resolves_unknown_to_general() -> None:
    registry = SessionPolicyRegistry()
    resolved = registry.resolve("unknown")
    assert resolved.category == "general"
    assert resolved == DEFAULT_RETENTION_POLICIES["general"]


def test_registry_register_overrides_existing() -> None:
    registry = SessionPolicyRegistry()
    new_policy = RetentionPolicy(category="timeline", ttl=timedelta(hours=1))
    registry.register(new_policy)
    assert registry.resolve("timeline").ttl == timedelta(hours=1)
