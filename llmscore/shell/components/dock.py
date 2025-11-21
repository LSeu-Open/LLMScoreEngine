"""Phase 2 dock component for the llmscore shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...utils.rich_render import RenderOptions, render_panel


@dataclass(slots=True)
class DockConfig:
    """User preferences for the interactive dock."""

    position: str = "bottom"
    show_status_chip: bool = True
    show_quick_actions: bool = True
    multiline_prompt: bool = False
    color_blind_mode: bool = False


@dataclass(slots=True)
class DockState:
    """Runtime state tracked by the dock."""

    expanded: bool = True
    last_command: Optional[str] = None
    latency_ms: Optional[float] = None
    profile: Optional[str] = None
    session_id: Optional[str] = None
    quick_actions: List[str] = field(default_factory=list)


class Dock:
    """Lightweight presenter for the Phase 2 command dock."""

    def __init__(self, config: DockConfig | None = None) -> None:
        self.config = config or DockConfig()
        self.state = DockState()
        self._palette = self._build_palette()

    def toggle(self) -> None:
        self.state.expanded = not self.state.expanded

    def snapshot(self) -> str:
        """Return a textual summary used for early integration/testing."""

        status = [
            f"Dock position: {self.config.position}",
            f"Expanded: {'yes' if self.state.expanded else 'no'}",
        ]
        if self.config.show_status_chip:
            bits = [self.state.profile or "default", self.state.session_id or "default"]
            status.append("Status chip: " + " / ".join(bits))
        if self.state.last_command:
            status.append(f"Last command: {self.state.last_command}")
        if self.state.latency_ms is not None:
            status.append(f"Latency: {self.state.latency_ms:.1f} ms")
        if self.config.show_quick_actions and self.state.quick_actions:
            actions = ", ".join(self.state.quick_actions[:3])
            status.append(f"Quick actions: {actions}")
        return "\n".join(status)

    def render(self, *, options: RenderOptions | None = None) -> None:
        """Render the dock placeholder via Rich panels."""

        title = "Dock" if self.state.expanded else "Dock (collapsed)"
        palette = self._palette["default"].copy()
        if options and options.color_system:
            palette.update(self._palette.get(options.color_system, {}))
        border_style = palette.get("border", "cyan")
        snapshot = self.snapshot()
        if options and options.reduced_motion:
            snapshot += "\n(reduced motion)"
        render_panel(snapshot, title=title, border_style=border_style, options=options)

    def _build_palette(self) -> Dict[str, Dict[str, str]]:
        default = {"border": "cyan"}
        color_blind = {"border": "bright_white"}
        return {
            "default": default,
            "standard": color_blind if self.config.color_blind_mode else default,
        }


__all__ = ["Dock", "DockConfig", "DockState"]
