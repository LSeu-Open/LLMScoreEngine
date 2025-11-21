"""Feature-flagged layout manager for the llmscore shell.

Phase 2 introduces a richer, multi-pane layout that can be toggled on via
configuration without disturbing the existing runtime flow. The layout module
provides light scaffolding so panes (timeline, context, command stream, dock)
can be orchestrated centrally while we incrementally wire in virtualization
and responsive behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..utils.rich_render import get_console, render_panel

PaneRenderer = Callable[[], None]


@dataclass(slots=True)
class LayoutConfig:
    """User-facing toggles for the new layout system."""

    enabled: bool = False
    timeline_position: str = "left"
    context_position: str = "right"
    dock_position: str = "bottom"


@dataclass(slots=True)
class PaneState:
    """Tracks visibility and optional sizing metadata for a pane."""

    visible: bool = True
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass(slots=True)
class FrameDimensions:
    width: int = 100
    height: int = 30


class LayoutManager:
    """Coordinates rendering of the feature-flagged multi-pane workspace."""

    def __init__(
        self,
        *,
        config: LayoutConfig | None = None,
        command_stream: PaneRenderer | None = None,
        timeline: PaneRenderer | None = None,
        context: PaneRenderer | None = None,
        dock: PaneRenderer | None = None,
    ) -> None:
        self.config = config or LayoutConfig()
        self._console = get_console()
        self._command_stream = command_stream
        self._timeline = timeline
        self._context = context
        self._dock = dock
        self._timeline_state = PaneState(visible=True)
        self._context_state = PaneState(visible=True)
        self._dock_state = PaneState(visible=True)
        self._frame = FrameDimensions()
        self._history_limit = 20
        self._stream_limit = 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(self) -> None:
        """Render the layout according to the current feature flag."""

        if not self.config.enabled:
            self._render_legacy()
            return
        self._render_multi_pane()

    def update_states(
        self,
        *,
        timeline: Optional[bool] = None,
        context: Optional[bool] = None,
        dock: Optional[bool] = None,
    ) -> None:
        if timeline is not None:
            self._timeline_state.visible = timeline
        if context is not None:
            self._context_state.visible = context
        if dock is not None:
            self._dock_state.visible = dock

    def summary(self) -> str:
        return "\n".join(
            [
                "Layout: experimental mode enabled"
                if self.config.enabled
                else "Layout: legacy",
                f"Timeline: {'visible' if self._timeline_state.visible else 'hidden'}",
                f"Context: {'visible' if self._context_state.visible else 'hidden'}",
                f"Dock: {'visible' if self._dock_state.visible else 'hidden'}",
            ]
        )

    @property
    def stream_limit(self) -> int:
        return self._stream_limit

    def set_frame(self, width: int, height: int) -> None:
        self._frame = FrameDimensions(width=width, height=height)

    def set_history_limit(self, limit: int) -> None:
        self._history_limit = max(1, limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _render_legacy(self) -> None:
        if self._command_stream:
            self._command_stream()
        if self._timeline_state.visible and self._timeline:
            self._timeline()
        if self._context_state.visible and self._context:
            self._context()
        if self._dock_state.visible and self._dock:
            self._dock()

    def _render_multi_pane(self) -> None:
        """Very small stub that will evolve into full pane composition."""

        render_panel(self.summary(), title="Phase 2 Layout")

        self._stream_limit = self._compute_stream_limit()
        if self._timeline_state.visible and self._timeline:
            self._timeline()
        if self._context_state.visible and self._context:
            self._context()
        if self._command_stream:
            self._command_stream()
        if self._dock_state.visible and self._dock:
            self._dock()

    def _compute_stream_limit(self) -> int:
        height = max(10, self._frame.height)
        available = max(3, height - 10)
        return max(3, min(self._history_limit, available))


__all__ = [
    "LayoutManager",
    "LayoutConfig",
    "PaneState",
    "FrameDimensions",
]
