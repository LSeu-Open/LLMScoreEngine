"""Tests for structured output cards and rich render helpers."""

from __future__ import annotations

import json

from llmscore.shell.components.output_cards import OutputCard
from llmscore.utils.rich_render import build_copy_hint, export_rows_to_csv


def test_output_card_from_payload_dict() -> None:
    payload = {"accuracy": 0.91, "loss": 0.12}
    card = OutputCard.from_payload(
        title="score.batch",
        payload=payload,
        summary="Batch scoring results",
        metadata={"Action": "score.batch"},
    )
    assert card.body == json.dumps(payload, indent=2)
    assert card.copy_hint and "Copy hint" in card.copy_hint
    assert card.summary == "Batch scoring results"


def test_output_card_render_exports_csv(tmp_path, monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _stub_panel(content, title, options=None, border_style=None):  # pragma: no cover - stub
        captured["content"] = content
        captured["title"] = title
        captured["border"] = border_style or ""

    monkeypatch.setattr(
        "llmscore.shell.components.output_cards.render_panel",
        _stub_panel,
    )
    rows = [{"model": "A", "score": 0.9}, {"model": "B", "score": 0.8}]
    card = OutputCard(
        title="Results",
        rows=rows,
        status="success",
        summary="Top models",
        metadata={"Action": "results.list"},
        copy_hint="Copy hint: ...",
    )
    export_path = tmp_path / "results.csv"
    card.render(export_path=export_path)
    assert export_path.exists()
    csv_content = export_path.read_text(encoding="utf-8")
    assert "model,score" in csv_content
    assert "A,0.9" in csv_content
    assert captured["title"] == "Results"
    assert captured["border"] == "green"


def test_export_rows_to_csv_handles_empty_rows(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    export_rows_to_csv(path, [])
    assert not path.exists()


def test_build_copy_hint_truncates_long_text() -> None:
    text = "line " * 30
    hint = build_copy_hint(text)
    assert hint.startswith("Copy hint")
