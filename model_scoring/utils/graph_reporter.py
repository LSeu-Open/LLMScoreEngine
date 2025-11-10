"""HTML reporting utilities for scoring results."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

DEFAULT_RESULTS_DIR = Path("Results")
DEFAULT_REPORTS_SUBDIR = "Reports"
DEFAULT_REPORT_FILENAME = "model_performance_report.html"
DEFAULT_REPORT_TITLE = "LLM Scoring Engine - Performance Report"


@dataclass(slots=True)
class SummaryData:
    """Key summary values rendered in the HTML report."""

    top_model_by_score: str = "N/A"
    model_count: int = 0
    generated_at: str = datetime.utcnow().isoformat()


def find_latest_csv_report(
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> Optional[Path]:
    """Return the most recent CSV report within the results directory."""

    reports_dir = results_dir / DEFAULT_REPORTS_SUBDIR
    if not reports_dir.exists():
        return None

    candidates = sorted(
        reports_dir.glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_report_data(csv_path: Path) -> pd.DataFrame:
    """Load the report CSV into a DataFrame."""

    return pd.read_csv(csv_path)


def build_summary(dataframe: pd.DataFrame) -> SummaryData:
    """Build summary metadata for the HTML template."""

    summary = SummaryData(model_count=len(dataframe.index))
    if dataframe.empty:
        return summary

    if "final_score" in dataframe.columns:
        top_index = dataframe["final_score"].astype(float).idxmax()
        top_row = dataframe.loc[top_index]
        summary.top_model_by_score = (
            f"{top_row.get('model_name', 'unknown')} "
            f"(Score: {float(top_row.get('final_score', 0.0)):.2f})"
        )
    return summary


def _resolve_template_env(template_dir: Optional[str]) -> Environment:
    search_path = (
        template_dir
        if template_dir is not None
        else Path(__file__).resolve().parent
    )
    return Environment(loader=FileSystemLoader(str(search_path)))


def render_html_report(
    dataframe: pd.DataFrame,
    summary: SummaryData,
    *,
    output_path: Path,
    template_dir: Optional[str] = None,
) -> Path:
    """Render the HTML report to disk using Jinja templates."""

    env = _resolve_template_env(template_dir)
    try:
        template = env.get_template("report_template.html")
    except TemplateNotFound:
        template = env.from_string(
            "<html><body><h1>{{ report_title }}</h1>"
            "<p>Generated at {{ summary_data.generated_at }}</p>"
            "<p>Top model: {{ summary_data.top_model_by_score }}</p>"
            "{{ table_html | safe }}</body></html>"
        )

    table_html = dataframe.to_html(index=False, classes="results-table")
    context: Dict[str, object] = {
        "report_title": DEFAULT_REPORT_TITLE,
        "summary_data": asdict(summary),
        "table_html": table_html,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.render(context), encoding="utf-8")
    return output_path


def generate_report(
    template_dir: Optional[str] = None,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    output_dir: Optional[Path | str] = None,
) -> Optional[str]:
    """Generate an HTML report from the latest CSV summary."""

    results_path = Path(results_dir)
    csv_path = find_latest_csv_report(results_path)
    if not csv_path:
        return None

    dataframe = load_report_data(csv_path)
    summary = build_summary(dataframe)

    destination_dir = (
        Path(output_dir)
        if output_dir is not None
        else results_path / DEFAULT_REPORTS_SUBDIR
    )
    output_path = destination_dir / DEFAULT_REPORT_FILENAME
    render_html_report(
        dataframe,
        summary,
        output_path=output_path,
        template_dir=template_dir,
    )
    return str(output_path)


__all__ = [
    "SummaryData",
    "build_summary",
    "find_latest_csv_report",
    "generate_report",
    "load_report_data",
    "render_html_report",
]
