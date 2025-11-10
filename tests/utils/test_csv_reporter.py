"""Tests for the csv_reporter module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from model_scoring.utils.csv_reporter import generate_csv_report


@pytest.fixture()
def test_environment(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary results and model directories with sample data."""

    results_dir = tmp_path / "Results"
    models_dir = tmp_path / "Models"
    reports_dir = results_dir / "Reports"

    results_dir.mkdir()
    models_dir.mkdir()
    reports_dir.mkdir()

    result_payload = {
        "model_name": "test-model-1",
        "scores": {
            "entity_score": 1.0,
            "dev_score": 2.0,
            "community_score": 3.0,
            "technical_score": 4.0,
            "final_score": 2.5,
        },
    }
    model_payload = {
        "model_specs": {
            "param_count": "10B",
            "architecture": "Transformer",
            "input_price": 0.001,
            "output_price": 0.002,
            "price": "Free",
        }
    }

    (results_dir / "test-model-1_results.json").write_text(
        json.dumps(result_payload),
        encoding="utf-8",
    )
    (models_dir / "test-model-1.json").write_text(
        json.dumps(model_payload),
        encoding="utf-8",
    )

    return results_dir, models_dir


def test_generate_csv_report_creates_expected_row(
    test_environment: tuple[Path, Path],
) -> None:
    """Ensure the generated CSV contains the expected single row."""

    results_dir, models_dir = test_environment

    report_path = generate_csv_report(
        models=["test-model-1"],
        results_dir=results_dir,
        models_dir=models_dir,
    )

    assert report_path.exists()

    with report_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["model_name"] == "test-model-1"
    assert row["param_count"] == "10B"
    assert row["architecture"] == "Transformer"
    assert row["price"] == "Free"
    assert float(row["entity_score"]) == 1.0
    assert float(row["dev_score"]) == 2.0
    assert float(row["community_score"]) == 3.0
    assert float(row["technical_score"]) == 4.0
    assert float(row["final_score"]) == 2.5
