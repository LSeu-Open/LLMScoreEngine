"""Helpers for asserting shell transcripts against golden baselines."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import yaml


@dataclass(slots=True)
class TranscriptEvent:
    prompt: str
    response: str


class TranscriptRecorder:
    def __init__(self, baselines_dir: Path) -> None:
        self.baselines_dir = baselines_dir

    def assert_match(self, flow_name: str, events: Iterable[TranscriptEvent]) -> None:
        payload = {"events": [asdict(event) for event in events]}
        baseline_path = self.baselines_dir / f"{flow_name}.yml"
        regen = os.getenv("TRANSCRIPTS_REGEN") == "1"
        if regen or not baseline_path.exists():
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            if regen:
                raise AssertionError(
                    f"Baseline regenerated for {flow_name}; verify and unset TRANSCRIPTS_REGEN"
                )
            raise AssertionError(
                f"Baseline created for {flow_name}. Review and re-run tests to lock it in."
            )

        baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
        assert baseline == payload, (
            f"Transcript mismatch for {flow_name}. To update, set TRANSCRIPTS_REGEN=1"
        )


__all__: List[str] = ["TranscriptRecorder", "TranscriptEvent"]
