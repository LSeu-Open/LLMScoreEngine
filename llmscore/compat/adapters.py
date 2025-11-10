"""Legacy command wrappers for integrating existing scripts with llmscore."""

from __future__ import annotations

from typing import Callable, Sequence


class LegacyCommandAdapter:
    """Wraps an imperative script entry point for use as an action handler."""

    def __init__(self, name: str, runner: Callable[..., int]) -> None:
        self._name = name
        self._runner = runner

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, argv: Sequence[str]) -> int:
        return self._runner(*argv)


__all__ = ["LegacyCommandAdapter"]
