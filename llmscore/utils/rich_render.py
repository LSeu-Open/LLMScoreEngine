"""Helpers for rendering rich output within the CLI."""

from __future__ import annotations

from typing import Any

from rich.console import Console


def get_console() -> Console:
    """Return a shared Rich console instance."""

    return Console()


def render_markdown(markdown: str) -> None:
    """Render markdown to the console."""

    console = get_console()
    console.print(markdown)


def render_panel(content: Any, title: str | None = None) -> None:
    """Render content inside a rich panel."""

    from rich.panel import Panel  # Local import to avoid eager dependency

    console = get_console()
    console.print(Panel.fit(content, title=title))


__all__ = ["get_console", "render_markdown", "render_panel"]
