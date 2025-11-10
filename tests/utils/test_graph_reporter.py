"""Tests for the graph_reporter module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from model_scoring.utils.graph_reporter import (
    DEFAULT_REPORT_FILENAME,
    generate_report,
)


@pytest.fixture()
def test_environment(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temporary results directory with a CSV and template."""

    results_dir = tmp_path / "Results"
    reports_dir = results_dir / "Reports"
    reports_dir.mkdir(parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = reports_dir / f"LLM-Scoring-Engine_report_{timestamp}.csv"
    data = {
        "model_name": ["test-model-1", "test-model-2"],
        "param_count": [10, 80],
        "architecture": ["Transformer", "MoE"],
        "final_score": [2.5, 3.5],
        "entity_score": [1.0, 3.0],
        "dev_score": [2.0, 2.0],
        "community_score": [3.0, 1.0],
        "technical_score": [4.0, 4.0],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "report_template.html").write_text(
        "<html><body><h1>{{ report_title }}</h1>"
        "<p>Top model: {{ summary_data.top_model_by_score }}</p>"
        "{{ table_html | safe }}</body></html>",
        encoding="utf-8",
    )

    return results_dir, template_dir


def test_generate_report_success(test_environment: tuple[Path, Path]) -> None:
    """Ensure generate_report renders the template with summary content."""

    results_dir, template_dir = test_environment
    output_dir = results_dir / "Reports" / "Rendered"

    report_path = generate_report(
        template_dir=str(template_dir),
        results_dir=results_dir,
        output_dir=output_dir,
    )

    expected_path = output_dir / DEFAULT_REPORT_FILENAME
    assert report_path == str(expected_path)
    assert expected_path.exists()

    html_content = expected_path.read_text(encoding="utf-8")
    assert "LLM Scoring Engine - Performance Report" in html_content
    assert "test-model-2 (Score: 3.50)" in html_content
