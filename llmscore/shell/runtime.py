"""Prompt-toolkit based runtime scaffold for the llmscore shell."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
import time
import shlex
from typing import Any, Dict, List, Optional, Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout

from ..actions.base import ActionExecutionError, ActionMetadata
from ..actions.catalog import register_default_actions
from ..actions.registry import ActionRegistry
from ..orchestrator.controller import TaskController
from ..orchestrator.events import EventBus, OrchestratorEvent
from ..state.store import SessionStore
from ..utils.logging import get_logger
from ..utils.rich_render import (
    RenderOptions,
    build_palette,
    get_console,
    render_panel,
    style_table,
)
from ..utils.perf_monitor import PerformanceBudget, PerformanceMonitor
from .components import Dock, DockConfig, OutputCard
from .commands.config import ConfigContext, run_guided_setup, show_configuration
from .layout import LayoutConfig, LayoutManager
from .palette import PaletteCompleter, PaletteEntry, PaletteProvider
from .scratchpad import ScratchpadManager
from .suggestions import SuggestionEngine
from .timeline import TimelineManager
from .preferences import apply_preferences, load_preferences

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

PERFORMANCE_DEFAULT_BUDGETS: Dict[str, tuple[float, str]] = {
    "command_loop": (120.0, "Single command dispatch end-to-end"),
    "command_stream_render": (60.0, "Command history table rendering"),
    "timeline_render": (45.0, "Timeline snapshot rendering"),
    "context_render": (45.0, "Context pane rendering"),
    "dock_render": (35.0, "Dock panel rendering"),
}


ASCII_BANNER = r"""
██╗     ██╗     ███╗   ███╗  ███████╗ ██████╗ ████████╗ ██████╗  ██████╗   ██████╗ ███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗
██║     ██║     ████╗ ████║  ██╔════╝██╔════╝║██╔═══██║██╔══██╗║██╔═════╝ ║██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝  
██║     ██║     ██╔████╔██║  ███████╗██║     ║██║   ██║██████╔╝║█████╗    ║█████╗  ██╔██╗ ██║██║  ███║██║██╔██╗ ██║█████╗   
██║     ██║     ██║╚██╔╝██║  ╚════██║██║     ║██║   ██║██╔══██╗║██╔══╝    ║██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  
███████╗███████╗██║ ╚═╝ ██║  ███████║╚██████║║████████║██║  ██║║███████║  ║███████║██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗         
╚══════╝╚══════╝╚═╝     ╚═╝  ╚══════╝ ╚═════╝╚════════╝╚═╝  ╚═╝╚═══════╝  ╚═══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝  
"""


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
    banner: Optional[str] = ASCII_BANNER
    enable_layout_v2: bool = False
    layout_timeline_position: str = "left"
    layout_context_position: str = "right"
    layout_dock_position: str = "bottom"
    use_output_cards: bool = False
    palette_autocomplete: bool = True
    reduced_motion: bool = False
    color_blind_mode: bool = False
    performance_monitor_enabled: bool = False
    performance_budgets: Optional[Dict[str, float]] = None


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
        self._render_options = RenderOptions(
            reduced_motion=self.config.reduced_motion,
            color_system="standard" if self.config.color_blind_mode else None,
            palette=build_palette(self.config.color_blind_mode),
        )
        self.console = get_console(self._render_options)
        self.logger = get_logger("llmscore.shell")
        self.event_bus = EventBus()
        self.controller = TaskController(event_bus=self.event_bus)
        self.session_store = session_store or SessionStore()
        self.preferences = load_preferences(self.session_store, self.config.profile)
        apply_preferences(self.session_store, self.preferences)

        if action_registry is None:
            self.action_registry = ActionRegistry(event_bus=self.event_bus)
            register_default_actions(self.action_registry)
        else:
            self.action_registry = action_registry
            self.action_registry.bind_event_bus(self.event_bus)

        self.event_bus.subscribe(self._handle_event)
        self.palette = palette_provider or self._bootstrap_palette()
        self._completer = (
            PaletteCompleter(self.palette)
            if self.config.palette_autocomplete
            else None
        )

        self.session_id = self.config.session_id or "default"
        self.profile = self.config.profile

        self.suggestions = SuggestionEngine(session_store=self.session_store)
        self.suggestions.bind_session(self.session_id, profile=self.profile)
        self.scratchpad = ScratchpadManager(session_store=self.session_store)
        self.scratchpad.bind_session(self.session_id, profile=self.profile)
        self.timeline = TimelineManager(session_store=self.session_store)
        self.timeline.bind_session(self.session_id, profile=self.profile)

        self.timeline_visible = self.config.enable_timeline
        self.context_visible = self.config.enable_context
        self.dock = Dock(
            DockConfig(color_blind_mode=self.config.color_blind_mode)
        )
        self.dock.state.profile = self.profile or "default"
        self.dock.state.session_id = self.session_id
        self._recent_commands: List[str] = []
        self._command_history_limit = 200
        layout_config = LayoutConfig(
            enabled=self.config.enable_layout_v2,
            timeline_position=self.config.layout_timeline_position,
            context_position=self.config.layout_context_position,
            dock_position=self.config.layout_dock_position,
        )
        self.layout_manager = LayoutManager(
            config=layout_config,
            command_stream=self._render_command_stream,
            timeline=lambda: self._render_timeline_snapshot(),
            context=self._render_context,
            dock=self._render_dock,
        )
        self.layout_manager.set_history_limit(20)
        self.layout_manager.update_states(
            timeline=self.timeline_visible,
            context=self.context_visible,
            dock=True,
        )
        self._refresh_dock_quick_actions()
        session_kwargs: Dict[str, Any] = {"history": InMemoryHistory()}
        if self._completer is not None:
            session_kwargs["completer"] = self._completer
            session_kwargs["complete_while_typing"] = True
        if self.config.palette_autocomplete:
            session_kwargs["bottom_toolbar"] = self._palette_toolbar
        self.session = PromptSession(**session_kwargs)
        self._bindings = self._build_key_bindings()
        self._running = False
        self._focus_order = ["timeline", "context", "command", "dock"]
        self._focus_index = 0
        self._perf_monitor = (
            self._build_performance_monitor()
            if self.config.performance_monitor_enabled
            else None
        )

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
            run_in_terminal(self.toggle_timeline)

        @bindings.add("c-c")
        def _(event) -> None:  # type: ignore[override]
            run_in_terminal(self.toggle_context)

        @bindings.add("c-k")
        def _(event) -> None:  # type: ignore[override]
            run_in_terminal(partial(self._open_palette, ""))

        @bindings.add("c-p")
        def _(event) -> None:  # type: ignore[override]
            run_in_terminal(self._cycle_focus)

        return bindings

    # ------------------------------------------------------------------
    # Command loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Enter the interactive loop until the user exits."""

        if self._running:
            return 0
        self._running = True
        self._render_intro()
        with patch_stdout():
            while self._running:
                try:
                    command = self.session.prompt(
                        self.config.prompt,
                        key_bindings=self._bindings,
                    )
                except KeyboardInterrupt:
                    self.console.print("^C")
                    continue
                except EOFError:
                    break
                command = command.strip()
                if not command:
                    continue
                with self._measure("command_loop"):
                    self._record_command(command)
                    try:
                        self._dispatch_command(command)
                    except Exception as exc:  # noqa: BLE001
                        self.logger.exception("Shell command failed")
                        self.console.print(f"[red]Command error:[/red] {exc}")
        self._running = False
        return 0

    def _dispatch_command(self, command: str) -> None:
        verb, *args = command.split()
        argument = " ".join(args)

        if verb in {"exit", "quit"}:
            self._running = False
            self.console.print("Exiting shell...")
            return
        if verb == "help":
            self._render_help()
            return
        if verb == "status":
            self._render_status()
            return
        if verb == "clear":
            self.console.clear()
            return
        if verb == "actions":
            self._render_actions()
            return
        if verb == "timeline":
            self.toggle_timeline()
            return
        if verb == "context":
            self.toggle_context()
            return
        if verb == "palette":
            self._open_palette(argument)
            return
        if verb == "suggest":
            self._render_suggestions()
            return
        if verb == "scratch":
            self._handle_scratch_command(argument)
            return
        if verb == "performance":
            self._render_performance_report()
            return
        if verb == "run" and argument:
            self._execute_action(argument)
            return
        if verb == "wizard" and argument:
            self.console.print(
                "Guided wizards are not available yet. Use 'run <action>' instead."
            )
            return
        if verb == "config":
            self._handle_config_command(argument)
            return

        # Try to resolve as an implicit action call
        if verb in self.action_registry:
            self._execute_action(command)
            return

        self.console.print(f"Unknown command: {command}")

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def toggle_timeline(self) -> None:
        self.timeline_visible = not self.timeline_visible
        state = "visible" if self.timeline_visible else "hidden"
        self.layout_manager.update_states(timeline=self.timeline_visible)
        self.console.print(f"Timeline pane is now {state}.")
        if self.timeline_visible:
            self._render_timeline_snapshot()
        self._render_layout_if_enabled()

    def toggle_context(self) -> None:
        self.context_visible = not self.context_visible
        state = "visible" if self.context_visible else "hidden"
        self.layout_manager.update_states(context=self.context_visible)
        self.console.print(f"Context pane is now {state}.")
        if self.context_visible:
            self._render_context()
        self._render_layout_if_enabled()

    def _cycle_focus(self) -> None:
        self._focus_index = (self._focus_index + 1) % len(self._focus_order)
        pane = self._focus_order[self._focus_index]
        self.console.print(f"[bold cyan]Focus[/bold cyan] → {pane.title()}")
        if pane == "timeline" and self.timeline_visible:
            self._render_timeline_snapshot()
        elif pane == "context" and self.context_visible:
            self._render_context()
        elif pane == "command":
            self.console.print("Command input ready.")
        elif pane == "dock":
            self._render_dock()

    def _open_palette(self, query: str) -> None:
        query = query.strip()
        entries = (
            list(self.palette.search(query, limit=5)) if query else list(self.palette.recommend(limit=5))
        )
        if not entries:
            self.console.print("No palette entries match your query.")
            return
        lines = []
        for entry in entries:
            desc = entry.description or ""
            lines.append(f"{entry.action:<20} {desc}")
        self._render_panel("\n".join(lines), title="Palette")

    def _handle_scratch_command(self, args: str) -> None:
        tokens = args.split(maxsplit=1)
        if not tokens:
            self.console.print("Usage: scratch [add|show|remove|clear] ...")
            return
        action = tokens[0]
        payload = tokens[1] if len(tokens) > 1 else ""
        if action == "add" and payload:
            self.scratchpad.add(payload)
            self.console.print("Added scratchpad note.")
            return
        if action == "show":
            self._render_scratchpad()
            return
        if action == "remove" and payload:
            self.scratchpad.remove(payload)
            self.console.print(f"Removed scratchpad note {payload}.")
            return
        if action == "clear":
            self.scratchpad.clear()
            self.console.print("Scratchpad cleared.")
            return
        self.console.print("Invalid scratch command. Try add/show/remove/clear.")

    def _render_actions(self) -> None:
        entries = [definition.metadata for definition in self.action_registry]
        if not entries:
            self.console.print("No actions registered.")
            return
        from rich.table import Table

        table = Table(title="Registered Actions")
        table.add_column("Name")
        table.add_column("Title")
        table.add_column("Domain")
        style_table(table, options=self._render_options)
        for metadata in entries:
            table.add_row(metadata.name, metadata.title, metadata.domain)
        self.console.print(table)
    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_intro(self) -> None:
        if self.config.banner:
            self.console.print(self.config.banner, style="bold cyan")
            self.console.print()
        self.console.rule("llmscore shell")
        self.console.print(self.config.intro_message)
        self.console.print("Type 'help' to see available commands.\n")
        self._render_status()
        self._render_layout_if_enabled()

    def _render_status(self) -> None:
        if self.layout_manager.config.enabled:
            self.layout_manager.update_states(
                timeline=self.timeline_visible,
                context=self.context_visible,
            )
            width = getattr(self.console, "width", 100) or 100
            height = getattr(self.console, "height", 30) or 30
            self.layout_manager.set_frame(width, height)
            self._render_panel(
                self.layout_manager.summary(),
                title="Layout",
            )
            return
        timeline_state = "visible" if self.timeline_visible else "hidden"
        context_state = "visible" if self.context_visible else "hidden"
        summary = (
            f"Timeline pane: {timeline_state}\n"
            f"Context pane: {context_state}\n"
            f"Active session: {self.session_id}"
        )
        self._render_panel(summary, title="Layout")

    def _render_command_stream(self) -> None:
        with self._measure("command_stream_render"):
            limit = 10
            if self.layout_manager.config.enabled:
                limit = self.layout_manager.stream_limit
            history = self._recent_commands[-limit:]
            if not history:
                self.console.print("No commands executed yet.")
                return
            from rich.table import Table

            table = Table(title="Command Stream", show_lines=False)
            table.add_column("#", justify="right", width=4)
            table.add_column("Command", overflow="fold")
            style_table(table, options=self._render_options)
            start = len(self._recent_commands) - len(history) + 1
            for offset, command in enumerate(history, start=start):
                table.add_row(str(offset), command)
            self.console.print(table)

    def _render_timeline_snapshot(self, limit: int = 5) -> None:
        with self._measure("timeline_render"):
            events = list(self.timeline.recent(limit))
            if not events:
                self.console.print("Timeline is empty.")
                return
            if self._render_options.reduced_motion:
                lines = [
                    f"{event.timestamp.strftime('%H:%M:%S')} {event.kind}: {event.message}"
                    for event in events[-limit:]
                ]
                self._render_panel("\n".join(lines), title="Timeline")
                return
            from rich.table import Table

            table = Table(title="Timeline (latest events)")
            table.add_column("Time")
            table.add_column("Kind")
            table.add_column("Message")
            style_table(table, options=self._render_options)
            for event in events[-limit:]:
                table.add_row(
                    event.timestamp.strftime("%H:%M:%S"),
                    event.kind,
                    event.message,
                )
            self.console.print(table)

    def _render_layout_if_enabled(self) -> None:
        if not self.layout_manager.config.enabled:
            return
        width = getattr(self.console, "width", 100) or 100
        height = getattr(self.console, "height", 30) or 30
        self.layout_manager.set_frame(width, height)
        self.layout_manager.render()

    def _record_command(self, command: str) -> None:
        command = command.strip()
        if not command:
            return
        self._recent_commands.append(command)
        if len(self._recent_commands) > self._command_history_limit:
            self._recent_commands = self._recent_commands[-self._command_history_limit :]

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
            "performance          Show performance budgets & samples",
            "config [cmd]         Configure retention policies",
            "status               Show pane visibility",
            "session branch <id>  Switch to a new session context",
            "clear                Clear the console",
            "help action <name>   Show detailed information for an action",
            "Ctrl+P               Cycle focus between panes",
        ]
        self._render_panel("\n".join(help_lines), title="Commands")

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
        style_table(table, options=self._render_options)
        for suggestion in suggestions:
            table.add_row(
                suggestion.title,
                suggestion.command,
                suggestion.reason,
            )
        self.console.print(table)

    def _render_context(self) -> None:
        with self._measure("context_render"):
            if self._render_options.reduced_motion:
                self.console.print("Context snapshot (reduced motion)")
            else:
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
        style_table(table, options=self._render_options)
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

    def _render_panel(
        self,
        content: Any,
        *,
        title: str | None = None,
    ) -> None:
        render_panel(
            content,
            title=title,
            options=self._render_options,
        )

    def _render_dock(self) -> None:
        with self._measure("dock_render"):
            self.dock.render(options=self._render_options)

    def _refresh_dock_quick_actions(self, focus_action: str | None = None) -> None:
        quick: List[str] = []
        hints = FOLLOW_UP_HINTS.get(focus_action or "", ())
        for hint in hints:
            quick.append(hint["command"])
        if len(quick) < 3:
            for entry in self.palette.recommend(limit=5):
                cmd = entry.action
                if cmd not in quick:
                    quick.append(cmd)
                if len(quick) >= 3:
                    break
        self.dock.state.quick_actions = quick[:3]

    def _palette_toolbar(self) -> HTML | str:
        if not self.config.palette_autocomplete:
            return ""
        buffer = self.session.default_buffer
        text = buffer.document.text_before_cursor.strip()
        if text:
            entries = list(self.palette.search(text, limit=3))
        else:
            entries = list(self.palette.recommend(limit=3))
        if not entries:
            return ""
        chips = []
        for entry in entries:
            desc = entry.description or entry.title
            chips.append(f"<b>{entry.action}</b> {desc}")
        content = "  |  ".join(chips)
        return HTML(f"<style bg='#262626' fg='#ffd866'> {content} </style>")

    def _execute_action(self, command_line: str) -> None:
        try:
            parts = shlex.split(command_line)
        except ValueError as e:
            self.console.print(f"[red]Error parsing command:[/red] {e}")
            return

        if not parts:
            return

        name = parts[0]
        args = parts[1:]
        inputs: Dict[str, Any] = {}

        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                inputs[key] = value
            else:
                # For now, we don't support positional arguments directly mapping to schema
                # unless we implement a more complex parser.
                # Warn the user or just ignore/add as flag?
                # Let's treat it as a boolean flag if valid, or just warn.
                self.console.print(
                    f"[yellow]Warning:[/yellow] Ignoring positional argument '{arg}'. "
                    "Please use key=value format."
                )

        if name not in self.action_registry:
            self.console.print(f"Unknown action: {name}")
            return

        definition = self.action_registry.get(name)
        valid_keys = set(definition.input_schema.keys()) if definition.input_schema else set()
        
        # Filter inputs to only include valid keys defined in the schema
        # This prevents passing arbitrary arguments to handlers that don't accept **kwargs
        filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
        
        # Warn about ignored arguments
        ignored_keys = set(inputs.keys()) - valid_keys
        if ignored_keys:
            self.console.print(
                f"[yellow]Warning:[/yellow] The following arguments are not supported by '{name}' and were ignored: {', '.join(ignored_keys)}"
            )

        start = time.perf_counter()
        try:
            result = self.action_registry.run(
                name, inputs=filtered_inputs, controller=self.controller
            )
        except Exception as exc:  # noqa: BLE001
            metadata = ActionMetadata(
                name=name,
                title=name,
                description=f"Execution failed for {name}",
                domain="unknown",
            )
            wrapped = ActionExecutionError(metadata, exc)
            event = OrchestratorEvent(
                kind="error",
                message=str(wrapped),
                timestamp=datetime.now(UTC),
            )
            self._handle_event(event)
            self.console.print(f"[red]{wrapped}[/red]")
            return

        duration = result.duration or (time.perf_counter() - start)
        result.duration = duration
        event = OrchestratorEvent(
            kind="info" if result.succeeded else "error",
            message=f"Action '{name}' completed in {duration:.2f}s",
            timestamp=datetime.now(UTC),
        )
        self._handle_event(event)
        self._refresh_dock_quick_actions(focus_action=name)

        if self.config.use_output_cards:
            self._render_action_output_card(result)
            return

        payload = result.output
        if payload is None:
            self.console.print(f"Action '{name}' completed.")
        else:
            formatted = self._pretty_format_payload(payload)
            self._render_panel(formatted, title=result.metadata.title)

    def _pretty_format_payload(self, payload: Any) -> Any:
        from rich.table import Table
        from rich import box
        
        if isinstance(payload, dict):
            # Use a minimal table structure for dictionaries
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Key", style="bold cyan")
            table.add_column("Value")
            
            for key, value in payload.items():
                if isinstance(value, dict):
                    # Recursive call for nested dictionaries (e.g. 'scores')
                    val_render = self._pretty_format_payload(value)
                elif isinstance(value, list):
                    if not value:
                        val_render = "-"
                    elif isinstance(value[0], (str, int, float, bool)):
                         val_render = ", ".join(str(v) for v in value)
                    elif isinstance(value[0], dict):
                         # Render list of dicts as a nested table
                         sub_table = Table(show_header=True, box=box.SIMPLE_HEADLESS, padding=(0, 1))
                         # Heuristic: Use first few keys as headers
                         headers = list(value[0].keys())[:4] 
                         for h in headers:
                             sub_table.add_column(h, style="cyan")
                         for item in value[:5]: # Limit rows for compactness
                             row = [str(item.get(h, "-")) for h in headers]
                             sub_table.add_row(*row)
                         if len(value) > 5:
                             sub_table.add_row(*["..." for _ in headers])
                         val_render = sub_table
                    else:
                         # Fallback for complex lists
                         import json
                         val_render = json.dumps(value, indent=2)
                elif isinstance(value, float):
                    val_render = f"{value:.4f}"
                elif value is None:
                    val_render = "-"
                else:
                    val_render = str(value)
                
                table.add_row(key, val_render)
            return table
            
        if isinstance(payload, list):
             # If the root output is a list of dicts
             if payload and isinstance(payload[0], dict):
                 table = Table(box=box.SIMPLE)
                 headers = list(payload[0].keys())
                 for h in headers:
                     table.add_column(h, style="bold cyan")
                 for item in payload:
                     row = [str(item.get(h, "")) for h in headers]
                     table.add_row(*row)
                 return table
             return ", ".join(str(v) for v in payload)

        return str(payload)

    def _render_action_output_card(self, result) -> None:
        payload = result.output
        metadata = {
            "Action": result.metadata.name,
            "Domain": result.metadata.domain,
            "Duration": f"{result.duration:.2f}s",
        }
        summary = result.metadata.description or result.metadata.title
        card = OutputCard.from_payload(
            title=result.metadata.title,
            payload=payload,
            status="success" if result.error is None else "error",
            summary=summary,
            metadata=metadata,
            reduced_motion=self.config.reduced_motion,
        )
        card.render()

    def _handle_event(self, event: OrchestratorEvent) -> None:
        self.timeline.append(event)
        if not self.timeline_visible:
            return
        label = {
            "error": "ERROR",
            "warning": "WARNING",
            "progress": "PROGRESS",
        }.get(event.kind, "INFO")
        timestamp = event.timestamp.strftime("%H:%M:%S")
        self.console.print(f"[{timestamp}] {label:<8} {event.message}")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _handle_config_command(self, args: str) -> None:
        subcommand = args.strip().lower() or "show"
        context = ConfigContext(
            store=self.session_store,
            preferences=self.preferences,
            suggestions=self.suggestions,
            scratchpad=self.scratchpad,
            timeline=self.timeline,
            profile=self.profile,
        )
        if subcommand in {"show"}:
            show_configuration(context, self.console)
            return
        if subcommand in {"setup", "wizard"}:
            run_guided_setup(context, self.console)
            return
        self.console.print("Usage: config [show|setup]")

    # ------------------------------------------------------------------
    # Performance helpers
    # ------------------------------------------------------------------
    def _build_performance_monitor(self) -> PerformanceMonitor:
        overrides = self.config.performance_budgets or {}
        budgets: Dict[str, PerformanceBudget] = {}
        for label, (default_threshold, description) in PERFORMANCE_DEFAULT_BUDGETS.items():
            threshold = overrides.get(label, default_threshold)
            budgets[label] = PerformanceBudget(
                label=label,
                threshold_ms=threshold,
                description=description,
            )
        for label, threshold in overrides.items():
            if label not in budgets:
                budgets[label] = PerformanceBudget(label=label, threshold_ms=threshold)
        return PerformanceMonitor(budgets=budgets, logger=self.logger)

    def _measure(self, label: str):
        if self._perf_monitor is None:
            return nullcontext()
        return self._perf_monitor.measure(label)

    def performance_summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        if self._perf_monitor is None:
            return {}
        return self._perf_monitor.summary()

    def _render_performance_report(self) -> None:
        if self._perf_monitor is None:
            self.console.print("Performance monitor disabled. Enable via config.")
            return
        summary = self._perf_monitor.summary()
        if not summary:
            self.console.print("No performance samples recorded yet.")
            return
        lines = []
        for label, payload in summary.items():
            last_ms = payload.get("last_ms")
            budget_ms = payload.get("budget_ms")
            violations = payload.get("violations")
            lines.append(
                f"{label}: {last_ms:.2f} ms"
                + (f" / budget {budget_ms:.2f} ms" if budget_ms else "")
                + (f" (violations: {violations})" if violations else "")
            )
        self._render_panel("\n".join(lines), title="Performance")

__all__ = ["ShellConfig", "ShellRuntime", "OutputCard", "patch_stdout"]
