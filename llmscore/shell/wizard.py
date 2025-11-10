"""Guided wizard scaffolding for interactive command flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..utils.logging import get_logger

LOGGER = get_logger("llmscore.shell.wizard")

InputCallback = Callable[[str, Optional[str]], str]
Transformer = Callable[[str], Any]
Validator = Callable[[Any], Optional[str]]
CompletionCallback = Callable[[Dict[str, Any]], None]


@dataclass(slots=True)
class WizardField:
    """Definition for a single input collected within a wizard step."""

    name: str
    prompt: str
    default: Optional[str] = None
    help_text: Optional[str] = None
    transform: Transformer = field(default=lambda value: value)
    validator: Optional[Validator] = None
    required: bool = True


@dataclass(slots=True)
class WizardStep:
    """A wizard step containing one or more fields."""

    title: str
    description: Optional[str] = None
    fields: List[WizardField] = field(default_factory=list)


@dataclass(slots=True)
class WizardResult:
    """Container for collected wizard data."""

    data: Dict[str, Any]
    step_data: Dict[str, Dict[str, Any]]


class Wizard:
    """Coordinates multi-step data collection with validation."""

    def __init__(
        self,
        *,
        steps: Iterable[WizardStep],
        on_complete: Optional[CompletionCallback] = None,
    ) -> None:
        self._steps = list(steps)
        self._on_complete = on_complete

    def run(self, ask: InputCallback) -> WizardResult:
        overall: Dict[str, Any] = {}
        per_step: Dict[str, Dict[str, Any]] = {}

        for index, step in enumerate(self._steps, start=1):
            LOGGER.debug("Wizard step %s: %s", index, step.title)
            step_payload: Dict[str, Any] = {}
            for field_def in step.fields:
                value = self._collect_field(field_def, ask)
                if value is None and field_def.required:
                    raise ValueError(f"Field '{field_def.name}' is required.")
                step_payload[field_def.name] = value
                overall[field_def.name] = value
            per_step[step.title] = step_payload

        result = WizardResult(data=overall, step_data=per_step)
        if self._on_complete:
            try:
                self._on_complete(result.data)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Wizard completion callback failed: %s", exc)
                raise
        return result

    def _collect_field(
        self,
        field_def: WizardField,
        ask: InputCallback,
    ) -> Any:
        prompt = field_def.prompt
        if field_def.help_text:
            prompt = f"{prompt}\n{field_def.help_text}"

        raw_value = ask(prompt, field_def.default)
        needs_value = (
            field_def.required and field_def.default is None
        )
        if raw_value == "" and needs_value:
            raise ValueError(f"Field '{field_def.name}' is required.")
        if raw_value == "" and field_def.default is not None:
            raw_value = field_def.default

        transformed: Any
        try:
            transformed = field_def.transform(raw_value)
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to parse '{field_def.name}': {exc}"
            raise ValueError(message) from exc

        if field_def.validator:
            error = field_def.validator(transformed)
            if error:
                raise ValueError(error)
        return transformed


__all__ = ["Wizard", "WizardField", "WizardStep", "WizardResult"]
