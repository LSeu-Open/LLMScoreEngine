"""Prompt-toolkit based runtime scaffold for the llmscore shell."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from ..actions.base import ActionExecutionError
from ..actions.catalog import register_default_actions
from ..actions.registry import ActionRegistry
from ..orchestrator.controller import TaskController
from ..orchestrator.events import EventBus, OrchestratorEvent
from ..state.store import SessionStore
from ..utils.logging import get_logger
from ..utils.rich_render import get_console, render_panel
from .palette import PaletteEntry, PaletteProvider
from .scratchpad import ScratchpadManager
from .suggestions import SuggestionEngine
from .wizard import Wizard, WizardField, WizardStep

FOLLOW_UP_HINTS: Dict[str, Sequence[Dict[str, str]]] = {
    "score.batch": (
        {
            "command": "score.report",
            "title": "Generate score report",
            "reason": "Summarise the latest batch run",
        },
        {
            "command": "results.leaderboard",
            "title": "Open leaderboard",
            "reason": "Compare recently scored models",
        },
    ),
    "results.pin": (
        {
            "command": "results.list",
            "title": "List saved results",
            "reason": "Review pinned artifacts",
        },
    ),
}


@dataclass(slots=True)
class ShellConfig:
    """Configuration for the interactive shell."""

    prompt: str = "llmscore> "
    enable_timeline: bool = True
    enable_context: bool = True
    intro_message: str = (
        "Welcome back! Press Ctrl+K for the palette, Ctrl+T for timeline, "
        "Ctrl+C for context, and type 'help' to learn more."
    )
    session_id: Optional[str] = None
    profile: Optional[str] = None
    palette_recent_limit: int = 20


class ShellRuntime:
    """Interactive runtime that mirrors the assistant shell experience."""

    def __init__(
        self,
        config: Optional[ShellConfig] = None,
        *,
        action_registry: Optional[ActionRegistry] = None,
        palette_provider: Optional[PaletteProvider] = None,
        session_store: Optional[SessionStore] = None,
    ) -> None:
        self.config = config or ShellConfig()
        self.console = get_console()
        self.logger = get_logger("llmscore.shell")
        self.event_bus = EventBus()
        self.controller = TaskController(event_bus=self.event_bus)
        self.session_store = session_store or SessionStore()

        if action_registry is None:
            self.action_registry = ActionRegistry(event_bus=self.event_bus)
            register_default_actions(self.action_registry)
        else:
            self.action_registry = action_registry
            self.action_registry.bind_event_bus(self.event_bus)

        self.event_bus.subscribe(self._handle_event)
        self.palette = palette_provider or self._bootstrap_palette()

        self.session_id = self.config.session_id or "default"
        self.profile = self.config.profile

        self.suggestions = SuggestionEngine(session_store=self.session_store)
        self.suggestions.bind_session(self.session_id, profile=self.profile)
        self.scratchpad = ScratchpadManager(session_store=self.session_store)
        self.scratchpad.bind_session(self.session_id, profile=self.profile)

        self.timeline_visible = self.config.enable_timeline
        self.context_visible = self.config.enable_context
        self.timeline_events: List[OrchestratorEvent] = []
        self.session = PromptSession(history=InMemoryHistory())
        self._bindings = self._build_key_bindings()
        self._running = False

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _bootstrap_palette(self) -> PaletteProvider:
        entries = [
            PaletteEntry(
                title=definition.metadata.title,
                action=definition.metadata.name,
                description=definition.metadata.description,
                tags=definition.metadata.tags,
            )
            for definition in self.action_registry
        ]
        return PaletteProvider(
            entries,
            recent_limit=self.config.palette_recent_limit,
        )

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("c-t")
        def _(event) -> None:  # type: ignore[override]
            event.app.run_in_terminal(self.toggle_timeline)

        @bindings.add("c-c")
        def _(event) -> None:  # type: ignore[override]
            event.app.run_in_terminal(self.toggle_context)

        @bindings.add("c-k")
        def _(event) -> None:  # type: ignore[override]
            event.app.run_in_terminal(partial(self._open_palette, ""))

        return bindings

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_intro(self) -> None:
        self.console.rule("[bold cyan]llmscore shell")
        self.console.print(self.config.intro_message)
        self.console.print("Type 'help' to see available commands.\n")
        self._render_status()

    def _render_status(self) -> None:
        timeline_state = "visible" if self.timeline_visible else "hidden"
        context_state = "visible" if self.context_visible else "hidden"
        summary = (
            f"Timeline pane: {timeline_state}\n"
            f"Context pane: {context_state}\n"
            f"Active session: {self.session_id}"
        )
        render_panel(summary, title="Layout")

    def _render_help(self) -> None:
        help_lines = [
            "help                 Show this message",
            "exit | quit          Leave the shell",
            "actions              List available actions",
            "run <action>         Execute a registered action",
            "wizard <action>      Launch guided form for an action",
            "palette [query]      Open the action palette",
            "suggest              Show contextual suggestions",
            "scratch [cmd]        Manage scratchpad (add/show/remove/clear)",
            "timeline             Toggle the timeline pane",
            "context              Toggle the context pane",
            "status               Show pane visibility",
            "session branch <id>  Switch to a new session context",
            "clear                Clear the console",
            "help action <name>   Show detailed information for an action",
        ]
        render_panel("\n".join(help_lines), title="Commands")

    def _render_suggestions(self) -> None:
        suggestions = self.suggestions.get_suggestions(limit=5)
        if not suggestions:
            self.console.print("No suggestions available yet.")
            return
        from rich.table import Table

        table = Table(title="Contextual Suggestions")
        table.add_column("Title")
        table.add_column("Command")
        table.add_column("Why")
        for suggestion in suggestions:
            table.add_row(
                suggestion.title,
                suggestion.command,
                suggestion.reason,
            )
        self.console.print(table)

    def _render_timeline_snapshot(self, limit: int = 5) -> None:
        if not self.timeline_events:
            self.console.print("Timeline is empty.")
            return
        from rich.table import Table

        latest = self.timeline_events[-limit:]
        table = Table(title="Timeline (latest events)")
        table.add_column("Time")
        table.add_column("Kind")
        table.add_column("Message")
        for event in latest:
            table.add_row(
                event.timestamp.strftime("%H:%M:%S"),
                event.kind,
                event.message,
            )
        self.console.print(table)

    def _render_context(self) -> None:
        self.console.rule("[bold cyan]Context")
        self._render_scratchpad()

    def _render_scratchpad(self) -> None:
        notes = list(self.scratchpad.list())
        if not notes:
            self.console.print(
                "Scratchpad is empty. Use 'scratch add' to capture notes."
            )
            return
        from rich.table import Table

        table = Table(title="Scratchpad")
        table.add_column("ID", style="cyan")
        table.add_column("Created")
        table.add_column("Tags")
        table.add_column("Content")
        for note in notes[:5]:
            tags = " ".join(note.tags)
            table.add_row(
                note.identifier[:8],
                note.created_at,
                tags,
                note.content,
            )
        if len(notes) > 5:
            table.caption = f"Showing 5 of {len(notes)} entries"
        self.console.print(table)

    def _open_palette(self, query: str) -> None:
        search_query = query.strip()
        matches = list(self.palette.search(search_query))
        if not matches:
            if search_query:
                message = f"No actions match '{search_query}'"
            else:
                message = "No actions registered."
            self.console.print(message)
            return
        from rich.table import Table

        table = Table(title="Action Palette")
        table.add_column("Title")
        table.add_column("Command")
        table.add_column("Description")
        for entry in matches[:10]:
            table.add_row(entry.title, entry.action, entry.description)
        self.console.print(table)

    # ------------------------------------------------------------------
    # Pane management
    # ------------------------------------------------------------------
    def toggle_timeline(self) -> None:
        self.timeline_visible = not self.timeline_visible
        state = "shown" if self.timeline_visible else "hidden"
        self.console.print(f"[cyan]Timeline pane {state}.[/]")
        self._render_status()
        if self.timeline_visible:
            self._render_timeline_snapshot()

    def toggle_context(self) -> None:
        self.context_visible = not self.context_visible
        state = "shown" if self.context_visible else "hidden"
        self.console.print(f"[cyan]Context pane {state}.[/]")
        self._render_status()
        if self.context_visible:
            self._render_context()

    def _render_action_catalog(self) -> None:
        from rich.table import Table

        table = Table(title="Registered Actions")
        table.add_column("Name", style="cyan")
        table.add_column("Title")
        table.add_column("Domain")
        table.add_column("Description")
        for definition in self.action_registry:
            table.add_row(
                definition.metadata.name,
                definition.metadata.title,
                definition.metadata.domain,
                definition.metadata.description,
            )
        self.console.print(table)

    def _render_action_help(self, name: str) -> None:
        from rich.table import Table

        try:
            definition = self.action_registry.get(name)
        except KeyError:
            self.console.print(f"[red]Unknown action:[/] {name}")
            return

        table = Table(title=f"Action: {definition.metadata.name}")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("Title", definition.metadata.title)
        table.add_row("Domain", definition.metadata.domain)
        table.add_row("Description", definition.metadata.description)
        table.add_row("Deprecated", str(definition.metadata.deprecated))
        if definition.metadata.tags:
            table.add_row("Tags", ", ".join(definition.metadata.tags))
        if definition.examples:
            table.add_row("Examples", "\n".join(definition.examples))

        if definition.input_schema:
            schema_table = Table(title="Inputs")
            schema_table.add_column("Name", style="cyan")
            schema_table.add_column("Schema")
            for key, value in definition.input_schema.items():
                schema_table.add_row(key, json.dumps(value, indent=2))
            self.console.print(table)
            self.console.print(schema_table)
        else:
            self.console.print(table)

    def _handle_event(self, event: OrchestratorEvent) -> None:
        self.timeline_events.append(event)
        if len(self.timeline_events) > 50:
            self.timeline_events = self.timeline_events[-50:]
        if not self.timeline_visible:
            return
        style = {
            "error": "[red]",
            "warning": "[yellow]",
            "progress": "[green]",
        }.get(event.kind, "[cyan]")
        self.console.print(f"{style}{event.message}[/]")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Run the interactive shell loop."""

        if self._running:
            self.logger.debug("Shell already running; ignoring duplicate call")
            return 0

        self._running = True
        self._render_intro()

        with patch_stdout():
            while self._running:
                try:
                    text = self.session.prompt(
                        self.config.prompt,
                        key_bindings=self._bindings,
                    )
                except EOFError:
                    self.console.print("\nExiting llmscore shell.")
                    break
                except KeyboardInterrupt:
                    self.console.print("\nUse 'exit' to leave the shell.")
                    continue

                command = text.strip()
                if not command:
                    continue

                if command in {"exit", "quit"}:
                    self.console.print("Goodbye!")
                    break
                if command in {"help", "?"}:
                    self._render_help()
                    continue
                if command == "status":
                    self._render_status()
                    if self.timeline_visible:
                        self._render_timeline_snapshot()
                    if self.context_visible:
                        self._render_context()
                    continue
                if command == "timeline":
                    self.toggle_timeline()
                    continue
                if command == "context":
                    self.toggle_context()
                    continue
                if command == "actions":
                    self._render_action_catalog()
                    continue
                if command.startswith("help action"):
                    _, _, action_name = command.partition("action")
                    self._render_action_help(action_name.strip())
                    continue
                if command.startswith("run "):
                    _, _, remainder = command.partition(" ")
                    self._execute_action(remainder)
                    continue
                if command.startswith("wizard "):
                    _, _, remainder = command.partition(" ")
                    self._run_wizard(remainder)
                    continue
                if command.startswith("palette"):
                    _, _, remainder = command.partition(" ")
                    self._open_palette(remainder)
                    continue
                if command == "suggest":
                    self._render_suggestions()
                    continue
                if command.startswith("scratch"):
                    _, _, remainder = command.partition(" ")
                    self._handle_scratch_command(remainder.strip())
                    continue
                if command.startswith("session branch"):
                    _, _, new_id = command.partition("branch")
                    self._branch_session(new_id.strip())
                    continue
                if command == "clear":
                    self.console.clear()
                    continue

                warning = (
                    f"[yellow]Unrecognized command:[/] {command}. "
                    "Type 'help' for assistance."
                )
                self.console.print(warning)

        self._running = False
        return 0

    def _execute_action(
        self,
        action: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        action_name = action.strip()
        if not action_name:
            self.console.print("[yellow]Usage: run <action-name>[/]")
            return
        try:
            result = self.action_registry.run(
                action_name,
                inputs=inputs,
                controller=self.controller,
            )
        except KeyError:
            self.console.print(f"[red]Unknown action:[/] {action_name}")
        except ActionExecutionError as exc:
            self.console.print(f"[red]{exc}[/]")
            self.suggestions.add_hint(
                command=f"help action {action_name}",
                title=f"Read docs for {action_name}",
                reason="Action failed",
                score=0.7,
            )
        else:
            self.console.print(
                f"[green]Action '{action_name}' completed in "
                f"{result.duration:.2f}s.[/]"
            )
            if result.output is not None:
                self.console.print(f"[green]Output:[/] {result.output}")
            self.palette.record_usage(action_name)
            self.suggestions.record_action(action_name)
            self._enqueue_followups(action_name)

    def _run_wizard(self, action: str) -> None:
        action_name = action.strip()
        if not action_name:
            self.console.print("[yellow]Usage: wizard <action-name>[/]")
            return
        try:
            definition = self.action_registry.get(action_name)
        except KeyError:
            self.console.print(f"[red]Unknown action:[/] {action_name}")
            return
        if not definition.input_schema:
            self.console.print(
                "[yellow]Action has no input schema; use 'run' instead.[/]"
            )
            return

        fields = [
            WizardField(
                name=key,
                prompt=value.get("description", key),
                default=self._schema_default(value),
                help_text=self._schema_help_text(value),
                transform=self._schema_transformer(value),
                validator=self._schema_validator(value),
                required=self._schema_required(value),
            )
            for key, value in definition.input_schema.items()
        ]
        wizard = Wizard(
            steps=[
                WizardStep(
                    title=definition.metadata.title,
                    fields=fields,
                )
            ]
        )

        def ask(prompt_text: str, default: Optional[str]) -> str:
            hint = f" [{default}]" if default else ""
            return self.session.prompt(f"{prompt_text}{hint}\n> ")

        try:
            result = wizard.run(ask)
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/]")
            return

        self._execute_action(action_name, inputs=result.data)

    # ------------------------------------------------------------------
    # Scratchpad helpers
    # ------------------------------------------------------------------
    def _handle_scratch_command(self, args: str) -> None:
        if not args or args == "show":
            self._render_scratchpad()
            return
        subcommand, _, remainder = args.partition(" ")
        subcommand = subcommand.lower()
        if subcommand == "add":
            content, tags = self._split_note_content(remainder)
            if not content:
                content = self.session.prompt("Scratchpad note> ")
            note = self.scratchpad.add(content, tags=tags)
            self.console.print(
                f"[green]Saved note {note.identifier[:8]}[/]"
            )
            if self.context_visible:
                self._render_scratchpad()
            return
        if subcommand == "remove":
            identifier = remainder.strip()
            if not identifier:
                self.console.print(
                    "[yellow]Usage: scratch remove <note-id>[/]"
                )
                return
            if self.scratchpad.remove(identifier):
                self.console.print(f"[green]Removed note {identifier}[/]")
            else:
                self.console.print(f"[yellow]Note not found:[/] {identifier}")
            return
        if subcommand == "clear":
            self.scratchpad.clear()
            self.console.print("[green]Scratchpad cleared.[/]")
            return
        if subcommand == "export":
            export = self.scratchpad.export()
            self.console.print(export or "Scratchpad is empty.")
            return
        self.console.print(
            "[yellow]Unrecognized scratchpad command. "
            "Use add/show/remove/clear/export.[/]"
        )

    def _split_note_content(self, raw: str) -> tuple[str, Sequence[str]]:
        raw = raw.strip()
        if not raw:
            return "", ()
        if " --tags=" not in raw:
            return raw, ()
        content, _, tag_part = raw.partition(" --tags=")
        tags = tuple(
            tag.strip()
            for tag in tag_part.split(",")
            if tag.strip()
        )
        return content.strip(), tags

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def _branch_session(self, new_session_id: str) -> None:
        if not new_session_id:
            self.console.print(
                "[yellow]Usage: session branch <new-session-id>[/]"
            )
            return
        self.session_id = new_session_id
        self.suggestions.bind_session(self.session_id, profile=self.profile)
        self.scratchpad.bind_session(self.session_id, profile=self.profile)
        self.console.print(
            f"[green]Session context switched to '{self.session_id}'.[/]"
        )
        if self.context_visible:
            self._render_context()

    # ------------------------------------------------------------------
    # Schema helpers for wizard scaffolding
    # ------------------------------------------------------------------
    @staticmethod
    def _schema_default(schema: Dict[str, Any]) -> Optional[str]:
        default = schema.get("default")
        if default is None:
            return None
        if isinstance(default, (list, dict)):
            return json.dumps(default)
        return str(default)

    @staticmethod
    def _schema_help_text(schema: Dict[str, Any]) -> Optional[str]:
        help_text = schema.get("help") or schema.get("tooltip")
        if schema.get("enum"):
            options = ", ".join(str(item) for item in schema["enum"])
            base = help_text or ""
            composed = f"{base}\nOptions: {options}"
            help_text = composed.strip()
        return help_text

    @staticmethod
    def _schema_required(schema: Dict[str, Any]) -> bool:
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            return "null" not in schema_type
        return schema_type != "null"

    def _schema_transformer(
        self,
        schema: Dict[str, Any],
    ) -> Callable[[str], Any]:
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            schema_type = next(
                (item for item in schema_type if item != "null"),
                "string",
            )

        def _transform(value: str) -> Any:
            if schema_type == "integer":
                return int(value)
            if schema_type == "number":
                return float(value)
            if schema_type == "boolean":
                lowered = value.lower()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
                raise ValueError("Expected boolean (true/false).")
            if schema_type == "array":
                return json.loads(value) if value else []
            if schema_type == "object":
                return json.loads(value) if value else {}
            return value

        return _transform

    def _schema_validator(
        self,
        schema: Dict[str, Any],
    ) -> Optional[Callable[[Any], Optional[str]]]:
        enum_values = schema.get("enum")
        if not enum_values:
            return None

        def _validate(value: Any) -> Optional[str]:
            if value not in enum_values:
                options = ", ".join(map(str, enum_values))
                return f"Value must be one of: {options}"
            return None

        return _validate

    # ------------------------------------------------------------------
    # Suggestion helpers
    # ------------------------------------------------------------------
    def _enqueue_followups(self, action_name: str) -> None:
        hints = FOLLOW_UP_HINTS.get(action_name, ())
        for hint in hints:
            self.suggestions.add_hint(
                command=hint["command"],
                title=hint["title"],
                reason=hint["reason"],
                score=0.8,
            )


__all__ = ["ShellConfig", "ShellRuntime"]
