"""Command-line entry point for the llmscore assistant."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer

from .actions.base import ActionExecutionError
from .actions.catalog import register_default_actions
from .actions.registry import ActionRegistry
from .shell.runtime import ShellRuntime
from .state.store import SessionStore
from .workflows.registry import WorkflowRegistry


app = typer.Typer(help="Unified assistant CLI for LLMScoreEngine")


def _build_registry() -> ActionRegistry:
    registry = ActionRegistry()
    register_default_actions(registry)
    return registry


def _coerce_value(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw


def _parse_param(param: str) -> Tuple[str, object]:
    if "=" not in param:
        raise ValueError(
            f"Parameters must be provided as key=value pairs: '{param}'"
        )
    key, value = param.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid parameter key in '{param}'")
    return key, _coerce_value(value.strip())


@app.command()
def shell() -> None:
    """Launch the interactive assistant shell."""

    runtime = ShellRuntime()
    runtime.run()


@app.command()
def run(
    workflow: Optional[str] = typer.Argument(
        None,
        help=(
            "Name of saved workflow to execute "
            "(omit when using automation flags)."
        ),
    ),
    params: List[str] = typer.Option(
        [],
        "--param",
        "-p",
        help=(
            "Override inputs using key=value pairs. Repeat for "
            "multiple values."
        ),
        show_default=False,
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Workflow registry profile to read from.",
    ),
    watch: List[Path] = typer.Option(
        [],
        "--watch",
        help=(
            "Watch a directory for changes and trigger an action "
            "(repeatable)."
        ),
        show_default=False,
    ),
    patterns: List[str] = typer.Option(
        [],
        "--pattern",
        help="Glob pattern to include when watching (repeatable).",
        show_default=False,
    ),
    ignore_patterns: List[str] = typer.Option(
        [],
        "--ignore-pattern",
        help="Glob pattern to ignore when watching (repeatable).",
        show_default=False,
    ),
    action: Optional[str] = typer.Option(
        None,
        "--action",
        help="Target action for automation modes (watch/schedule).",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help=(
            "Recurse into subdirectories when watching for changes."
        ),
    ),
    debounce: float = typer.Option(
        1.0,
        "--debounce",
        help="Debounce interval in seconds for watch mode.",
    ),
    cron: Optional[str] = typer.Option(
        None,
        "--cron",
        help="Cron expression to schedule an action (e.g. '0 2 * * *').",
    ),
    background: bool = typer.Option(
        False,
        "--background/--foreground",
        help="Run automation without blocking the terminal.",
    ),
    duration: Optional[float] = typer.Option(
        None,
        "--duration",
        help=(
            "For foreground scheduling, duration in seconds to keep "
            "running."
        ),
    ),
    webhook_url: Optional[str] = typer.Option(
        None,
        "--webhook",
        help="Webhook URL for automation notifications.",
    ),
    webhook_method: Optional[str] = typer.Option(
        None,
        "--webhook-method",
        help="HTTP method for webhook notifications.",
    ),
    webhook_payload_key: Optional[str] = typer.Option(
        None,
        "--webhook-payload-key",
        help="Payload key used when sending webhook notifications.",
    ),
    webhook_timeout: Optional[float] = typer.Option(
        None,
        "--webhook-timeout",
        help="Webhook request timeout in seconds.",
    ),
    webhook_headers: List[str] = typer.Option(
        [],
        "--webhook-header",
        help=(
            "Additional webhook header in key=value format (repeatable)."
        ),
        show_default=False,
    ),
    timezone_name: Optional[str] = typer.Option(
        None,
        "--timezone",
        help="Timezone identifier for cron scheduling (e.g. 'UTC').",
    ),
) -> None:
    """Execute saved workflows or automation helpers headlessly."""

    if watch and cron:
        typer.secho(
            "Cannot use --watch and --cron simultaneously.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    registry = _build_registry()

    param_overrides: Dict[str, object] = {}
    for param in params:
        key, value = _parse_param(param)
        param_overrides[key] = value

    header_map: Dict[str, str] = {}
    for header in webhook_headers:
        key, value = _parse_param(header)
        header_map[str(key)] = str(value)

    if watch:
        if not action:
            typer.secho(
                "--action is required when using --watch.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        watch_inputs: Dict[str, object] = {
            "paths": [str(path) for path in watch],
            "action": action,
            "inputs": param_overrides,
            "recursive": recursive,
            "debounce_seconds": debounce,
            "background": background,
        }
        if patterns:
            watch_inputs["patterns"] = patterns
        if ignore_patterns:
            watch_inputs["ignore_patterns"] = ignore_patterns
        if webhook_url:
            watch_inputs["webhook_url"] = webhook_url
        if webhook_method:
            watch_inputs["webhook_method"] = webhook_method
        if webhook_payload_key:
            watch_inputs["webhook_payload_key"] = webhook_payload_key
        if webhook_timeout is not None:
            watch_inputs["webhook_timeout"] = webhook_timeout
        if header_map:
            watch_inputs["webhook_headers"] = header_map
        result = registry.run("automation.watch", inputs=watch_inputs)
        if result.output is not None:
            typer.echo(json.dumps(result.output, indent=2, default=str))
        return

    if cron:
        if not action:
            typer.secho(
                "--action is required when using --cron.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        schedule_inputs: Dict[str, object] = {
            "action": action,
            "cron": cron,
            "inputs": param_overrides,
            "background": background,
        }
        if timezone_name:
            schedule_inputs["timezone"] = timezone_name
        if duration is not None:
            schedule_inputs["duration_seconds"] = duration
        if webhook_url:
            schedule_inputs["webhook_url"] = webhook_url
        if webhook_method:
            schedule_inputs["webhook_method"] = webhook_method
        if webhook_payload_key:
            schedule_inputs["webhook_payload_key"] = webhook_payload_key
        if webhook_timeout is not None:
            schedule_inputs["webhook_timeout"] = webhook_timeout
        if header_map:
            schedule_inputs["webhook_headers"] = header_map
        result = registry.run("automation.schedule", inputs=schedule_inputs)
        if result.output is not None:
            typer.echo(json.dumps(result.output, indent=2, default=str))
        return

    if not workflow:
        typer.secho(
            "Workflow name is required when not using automation flags.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    workflow_registry = WorkflowRegistry(SessionStore(), profile=profile)
    definition = workflow_registry.get(workflow)
    if not definition:
        typer.secho(
            f"Workflow '{workflow}' not found.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    registered_actions = set(registry.names())
    missing_actions = [
        name
        for name in definition.required_actions()
        if name not in registered_actions
    ]
    if missing_actions:
        typer.secho(
            "Required actions are not registered: "
            + ", ".join(sorted(missing_actions)),
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    global_overrides: Dict[str, object] = {}
    per_action_overrides: Dict[str, Dict[str, object]] = {}
    for key, value in param_overrides.items():
        if "." in key:
            action_name, nested_key = key.split(".", 1)
            target_overrides = per_action_overrides.setdefault(action_name, {})
            target_overrides[nested_key] = value
        else:
            global_overrides[key] = value

    step_results: List[Dict[str, object]] = []
    try:
        for step in definition.steps:
            step_inputs = dict(step.inputs)
            step_inputs.update(global_overrides)
            if step.action in per_action_overrides:
                step_inputs.update(per_action_overrides[step.action])
            result = registry.run(step.action, inputs=step_inputs)
            step_results.append(
                {
                    "action": step.action,
                    "description": step.description,
                    "output": result.output,
                    "duration": result.duration,
                }
            )
    except ActionExecutionError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    payload = {
        "workflow": definition.name,
        "profile": profile,
        "steps": step_results,
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def exec(  # noqa: A002 - intentional command name
    action: str,
    params: Optional[List[str]] = typer.Argument(
        None, metavar="PARAMS...", help="Action parameters as key=value"
    ),
) -> None:
    """Execute a single action in batch mode."""

    registry = _build_registry()
    try:
        parsed_inputs = {}
        for param in params or []:
            key, value = _parse_param(param)
            parsed_inputs[key] = value
        result = registry.run(action, inputs=parsed_inputs)
    except KeyError:
        typer.secho(
            f"Unknown action '{action}'.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None
    except ActionExecutionError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    output = result.output
    if output is None:
        typer.echo("null")
    else:
        typer.echo(json.dumps(output, indent=2, default=str))


def main() -> None:
    """Entry point compatible with console_scripts."""

    app()


if __name__ == "__main__":
    main()
