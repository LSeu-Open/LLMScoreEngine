"""HTML reporting utilities for scoring results."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import json

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
    env = Environment(loader=FileSystemLoader(str(search_path)))
    
    # Add escapejs filter for JavaScript-safe JSON embedding
    def escapejs_filter(value):
        """Escape a string for safe embedding in JavaScript."""
        if value is None:
            return ''
        # For JSON strings that will be parsed with JSON.parse(),
        # we need to escape the string delimiters but keep JSON intact
        if not isinstance(value, str):
            value = str(value)
        # Escape backslashes first, then single quotes
        # (JSON strings use double quotes internally, so we escape single quotes for the JS context)
        return value.replace('\\', '\\\\').replace("'", "\\'")
    
    env.filters['escapejs'] = escapejs_filter
    return env


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

    numeric_columns = {
        "param_count",
        "input_price",
        "output_price",
        "price",
        "entity_score",
        "dev_score",
        "community_score",
        "technical_score",
        "final_score",
    }

    sanitized_records = []
    for record in dataframe.to_dict(orient="records"):
        sanitized_record: Dict[str, object] = {}
        for key, value in record.items():
            if pd.isna(value):
                sanitized_record[key] = 0.0 if key in numeric_columns else ""
                continue
            if key in numeric_columns:
                try:
                    sanitized_record[key] = float(value)
                except (TypeError, ValueError):
                    sanitized_record[key] = 0.0
            else:
                sanitized_record[key] = value
        sanitized_records.append(sanitized_record)

    radar_chart_data = [
        {
            "model_name": record.get("model_name", "Unknown"),
            "entity_score": record.get("entity_score", 0.0),
            "dev_score": record.get("dev_score", 0.0),
            "community_score": record.get("community_score", 0.0),
            "technical_score": record.get("technical_score", 0.0),
            "final_score": record.get("final_score", 0.0),
        }
        for record in sanitized_records
    ]

    leaderboard_json = json.dumps(sanitized_records)
    radar_json = json.dumps(radar_chart_data)
    summary_json = json.dumps(asdict(summary))

    context: Dict[str, object] = {
        "report_title": DEFAULT_REPORT_TITLE,
        "summary_data": asdict(summary),
        "table_html": table_html,
        "intro_text": (
            "Explore the latest scoring results comparing entity, developer, "
            "community, and technical benchmarks."
        ),
        "timestamp": summary.generated_at,
        "leaderboard_headers": list(dataframe.columns),
        "leaderboard_data": sanitized_records,
        "leaderboard_json": leaderboard_json,
        "final_score_col_index": dataframe.columns.get_loc("final_score")
        if "final_score" in dataframe.columns
        else -1,
        "default_visible_cols": list(dataframe.columns[:6]),
        "min_param": float(dataframe.get("param_count", pd.Series([0])).min())
        if "param_count" in dataframe.columns
        else 0.0,
        "max_param": float(dataframe.get("param_count", pd.Series([0])).max())
        if "param_count" in dataframe.columns
        else 0.0,
        "fig_scatter_html": "",
        "fig_bar_faceted_html": "",
        "fig_pie_composition_html": "",
        "fig_cost_html": "",
        "fig_cost": bool(sanitized_records),
        "cost_efficiency_data": sanitized_records,
        "fig_violin_html": "",
        "radar_chart_data": radar_chart_data,
        "radar_json": radar_json,
        "fig_radar_html": "",
        "fig_bar_html": "",
        "summary_json": summary_json,
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
