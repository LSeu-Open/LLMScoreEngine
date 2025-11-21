"""Helpers for rendering rich output within the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import csv

from rich.console import Console

_CONSOLE: Console | None = None


@dataclass(slots=True)
class ColorPalette:
    """Color tokens used throughout the shell UI."""

    panel_border: str = "cyan"
    table_border: str = "cyan"
    table_header: str = "bold cyan"
    row_alt: str = "dim"
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"


def build_palette(color_blind_mode: bool = False) -> ColorPalette:
    if not color_blind_mode:
        return ColorPalette()
    return ColorPalette(
        panel_border="bright_white",
        table_border="bright_white",
        table_header="bold bright_white",
        row_alt="grey58",
        success="bright_cyan",
        warning="bright_yellow",
        error="bright_magenta",
    )


@dataclass(slots=True)
class RenderOptions:
    """Options influencing how rich helpers behave."""

    reduced_motion: bool = False
    color_system: str | None = None
    palette: ColorPalette = field(default_factory=ColorPalette)


def get_console(options: RenderOptions | None = None) -> Console:
    """Return a shared Rich console instance configured for plain output."""

    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(
            soft_wrap=True,
            color_system=options.color_system if options else None,
            markup=False,
            highlight=False,
        )
    return _CONSOLE


def render_markdown(markdown: str, *, options: RenderOptions | None = None) -> None:
    """Render markdown to the console."""

    console = get_console(options)
    console.print(markdown)


def render_panel(
    content: Any,
    title: str | None = None,
    *,
    options: RenderOptions | None = None,
    border_style: str | None = None,
) -> None:
    """Render content inside a rich panel."""

    from rich.panel import Panel  # Local import to avoid eager dependency

    console = get_console(options)
    resolved_border = border_style or (
        options.palette.panel_border if options else ""
    )
    console.print(Panel.fit(content, title=title, border_style=resolved_border))


def style_table(table: Any, *, options: RenderOptions | None = None) -> None:
    """Apply accessibility-aware styling to Rich tables."""

    palette = options.palette if options else ColorPalette()
    if hasattr(table, "border_style"):
        table.border_style = palette.table_border
    if hasattr(table, "header_style"):
        table.header_style = palette.table_header
    if hasattr(table, "row_styles"):
        table.row_styles = ("", palette.row_alt)


def export_rows_to_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write table rows to CSV and return the resulting path."""

    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: Sequence[str]
    sample = rows[0]
    if isinstance(sample, Mapping):
        fieldnames = list(sample.keys())
    else:
        fieldnames = [f"col_{idx}" for idx in range(len(sample))]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if isinstance(row, Mapping):
                writer.writerow({key: row.get(key) for key in fieldnames})
            else:
                writer.writerow({key: value for key, value in zip(fieldnames, row)})
    return path


def build_copy_hint(text: str) -> str:
    """Return a hint instructing the user how to copy text to clipboard."""

    truncated = text.strip().splitlines()
    preview = truncated[0][:60] + ("…" if len(truncated[0]) > 60 else "") if truncated else ""
    return f"Copy hint: echo '{preview}' | clip"


__all__ = [
    "ColorPalette",
    "RenderOptions",
    "build_palette",
    "get_console",
    "render_markdown",
    "render_panel",
    "export_rows_to_csv",
    "build_copy_hint",
]
