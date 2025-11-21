"""Top-level pytest configuration; re-export CLI fixtures globally."""
from __future__ import annotations

pytest_plugins = ["tests.cli.conftest"]
