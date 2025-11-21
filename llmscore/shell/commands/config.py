"""Interactive configuration command and guided setup for the shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table

from ...state.policies import SessionPolicyRegistry
from ...state.store import SessionStore
from ...utils.logging import get_logger
from ..preferences import (
    ShellPreferences,
    apply_preferences,
    save_preferences,
)
from ..suggestions import SuggestionEngine
from ..timeline import TimelineManager
from ..scratchpad import ScratchpadManager
from ..wizard import Wizard, WizardField, WizardResult, WizardStep

LOGGER = get_logger("llmscore.shell.commands.config")


@dataclass(slots=True)
class ConfigContext:
    """Container bundling state providers needed by the config command."""

    store: SessionStore
    preferences: ShellPreferences
    suggestions: SuggestionEngine
    scratchpad: ScratchpadManager
    timeline: TimelineManager
    profile: Optional[str]


def _render_retention_table(console: Console, registry: SessionPolicyRegistry) -> None:
    table = Table(title="Session Retention Policies")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("TTL")
    table.add_column("Max Entries")
    table.add_column("Protect New")
    for category, policy in sorted(registry.to_mapping().items()):
        ttl = "permanent" if not policy.ttl else str(policy.ttl)
        limit = policy.max_entries or "unbounded"
        protect = "yes" if policy.protect_new_records else "no"
        table.add_row(category, ttl, str(limit), protect)
    console.print(table)


def _retention_wizard(registry: SessionPolicyRegistry) -> Wizard:
    categories = list(registry.categories())
    step = WizardStep(
        title="Retention Settings",
        description=(
            "Configure retention policy overrides. Leave values blank to keep"
            " current defaults. TTL accepts durations like '14d' or '12h'."
        ),
        fields=[
            WizardField(
                name="category",
                prompt="Which category do you want to override?",
                default=categories[0] if categories else "general",
            ),
            WizardField(
                name="ttl",
                prompt="New TTL (e.g. 7d, 12h) or leave blank",
                required=False,
            ),
            WizardField(
                name="max_entries",
                prompt="Maximum entries (leave blank for unlimited)",
                required=False,
            ),
            WizardField(
                name="protect_new_records",
                prompt="Protect newly persisted records? (yes/no)",
                required=False,
            ),
        ],
    )
    return Wizard(steps=[step])


def _guided_setup(context: ConfigContext, console: Console) -> None:
    active_registry = context.store.get_policy_registry()
    wizard = _retention_wizard(active_registry)

    def ask(prompt: str, default: Optional[str]) -> str:
        suffix = f" [{default}]" if default else ""
        return console.input(f"{prompt}{suffix}\n> ")

    try:
        result = wizard.run(ask)
    except ValueError as exc:
        console.print(f"[red]Setup aborted:[/] {exc}")
        return

    overrides = _coerce_retention_overrides(result)
    if not overrides:
        console.print("No retention changes applied.")
        return

    context.preferences.retention_overrides.update(overrides)
    save_preferences(context.store, context.profile, context.preferences)
    registry = apply_preferences(context.store, context.preferences)
    console.print("Updated retention settings:")
    console.print_json(data=context.preferences.retention_overrides)
    _render_retention_table(console, registry)


def _coerce_retention_overrides(result: WizardResult) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    category = result.data.get("category")
    if isinstance(category, str):
        payload: Dict[str, Any] = {}
        ttl = result.data.get("ttl")
        if isinstance(ttl, str) and ttl.strip():
            payload["ttl"] = ttl.strip()
        max_entries = result.data.get("max_entries")
        if isinstance(max_entries, str) and max_entries.strip():
            try:
                payload["max_entries"] = int(max_entries)
            except ValueError:
                LOGGER.warning("Invalid max_entries value '%s' ignored", max_entries)
        protect = result.data.get("protect_new_records")
        if isinstance(protect, str) and protect.strip():
            lowered = protect.strip().lower()
            payload["protect_new_records"] = lowered in {"true", "yes", "1"}
        if payload:
            overrides[category] = payload
    return overrides


def show_configuration(context: ConfigContext, console: Console) -> None:
    console.rule("Shell Configuration Overview")
    console.print(f"Active profile: {context.profile or 'default'}")
    if context.preferences.updated_at:
        console.print(f"Preferences last updated: {context.preferences.updated_at}")
    registry = context.store.get_policy_registry()
    _render_retention_table(console, registry)
    if context.preferences.retention_overrides:
        console.print("\nActive overrides:")
        console.print_json(data=context.preferences.retention_overrides)


def run_guided_setup(context: ConfigContext, console: Console) -> None:
    console.rule("Interactive Shell Setup")
    _guided_setup(context, console)


__all__ = [
    "ConfigContext",
    "show_configuration",
    "run_guided_setup",
]
