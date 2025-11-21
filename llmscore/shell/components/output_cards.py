"""Structured output card rendering for shell actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...utils.rich_render import (
    RenderOptions,
    build_copy_hint,
    export_rows_to_csv,
    render_panel,
)


@dataclass(slots=True)
class OutputCard:
    """Structured rendering for action results."""

    title: str
    status: str = "success"
    summary: str | None = None
    metadata: Mapping[str, Any] | None = None
    body: str | None = None
    rows: Sequence[Mapping[str, Any]] | None = None
    copy_hint: str | None = None
    reduced_motion: bool = False
    footer: str | None = None

    def render(self, *, export_path: Path | None = None) -> None:
        """Render the output card, optionally exporting rows to CSV."""

        content_parts: list[str] = []
        if self.summary:
            content_parts.append(self.summary)
        if self.metadata:
            meta_text = "\n".join(
                f"{key}: {value}" for key, value in self.metadata.items()
            )
            content_parts.append(meta_text)
        if self.body:
            content_parts.append(self.body)
        if self.rows and export_path:
            export_rows_to_csv(export_path, self.rows)
            content_parts.append(f"CSV exported to {export_path}")
        if self.copy_hint:
            content_parts.append(self.copy_hint)
        if self.footer:
            content_parts.append(self.footer)
        content = "\n\n".join(content_parts)
        options = RenderOptions(reduced_motion=self.reduced_motion)
        border = {
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }.get(self.status, "cyan")
        render_panel(content, title=self.title, options=options, border_style=border)

    @classmethod
    def from_payload(
        cls,
        *,
        title: str,
        payload: Any,
        status: str = "success",
        summary: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        reduced_motion: bool = False,
    ) -> "OutputCard":
        body: str | None = None
        rows: Sequence[Mapping[str, Any]] | None = None
        copy_hint: str | None = None
        if isinstance(payload, Mapping):
            body = json.dumps(payload, indent=2)
            copy_hint = build_copy_hint(body)
        elif (
            isinstance(payload, Sequence)
            and payload
            and isinstance(payload[0], Mapping)
        ):
            rows = list(payload)  # type: ignore[list-item]
            preview = json.dumps(rows[:1], indent=2)
            copy_hint = build_copy_hint(preview)
        elif payload is not None:
            body = str(payload)
            copy_hint = build_copy_hint(body)
        footer = f"Rendered at {datetime.now(UTC).isoformat()}"
        card = cls(
            title=title,
            status=status,
            summary=summary,
            metadata=metadata,
            body=body,
            rows=rows,
            copy_hint=copy_hint,
            reduced_motion=reduced_motion,
            footer=footer,
        )
        return card


__all__ = ["OutputCard"]
