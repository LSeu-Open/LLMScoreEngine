# ------------------------------------------------------------------------------------------------
# License
# ------------------------------------------------------------------------------------------------

# Copyright (c) 2025 LSeu-Open
# 
# This code is licensed under the MIT License.
# See LICENSE file in the root directory

# ------------------------------------------------------------------------------------------------
# Description
# ------------------------------------------------------------------------------------------------

"""Utilities for exporting scoring results to CSV reports."""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from __future__ import annotations
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_RESULTS_DIR = Path("Results")
DEFAULT_MODELS_DIR = Path("Models")
DEFAULT_REPORTS_SUBDIR = "Reports"
DEFAULT_PROJECT_NAME = "LLM-Scoring-Engine"

LOGGER = logging.getLogger(__name__)

DEFAULT_HEADERS: Sequence[str] = (
    "model_name",
    "param_count",
    "architecture",
    "input_price",
    "output_price",
    "price",
    "entity_score",
    "dev_score",
    "community_score",
    "technical_score",
    "final_score",
)


@dataclass(slots=True)
class ReportRow:
    """Normalized representation of a single report row."""

    model_name: str
    param_count: Optional[str]
    architecture: Optional[str]
    input_price: Optional[float]
    output_price: Optional[float]
    price: Optional[float]
    entity_score: Optional[float]
    dev_score: Optional[float]
    community_score: Optional[float]
    technical_score: Optional[float]
    final_score: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "param_count": self.param_count,
            "architecture": self.architecture,
            "input_price": self.input_price,
            "output_price": self.output_price,
            "price": self.price,
            "entity_score": self.entity_score,
            "dev_score": self.dev_score,
            "community_score": self.community_score,
            "technical_score": self.technical_score,
            "final_score": self.final_score,
        }


def _load_json(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        LOGGER.debug("JSON file not found: %s", path)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to decode JSON %s: %s", path, exc)
    except OSError as exc:
        LOGGER.warning("Failed to read file %s: %s", path, exc)
    return None


def _model_name_from_result(path: Path, payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("model_name"), str):
        return payload["model_name"]
    return path.name.replace("_results.json", "")


def _collect_result_files(
    results_dir: Path,
    models: Optional[Sequence[str]] = None,
) -> List[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if models:
        requested = []
        for model in models:
            requested.append(results_dir / f"{model}_results.json")
        return [path for path in requested if path.exists()]

    return sorted(results_dir.glob("*_results.json"))


def _build_row(
    result_path: Path,
    result_payload: Mapping[str, Any],
    models_dir: Path,
) -> ReportRow:
    model_name = _model_name_from_result(result_path, result_payload)
    
    # Prefer model_specs from the result payload if available
    input_data = result_payload.get("input_data", {})
    model_specs = input_data.get("model_specs", {})
    
    # Fallback to loading from models_dir if not in result payload
    if not model_specs:
        model_payload = _load_json(models_dir / f"{model_name}.json") or {}
        model_specs = model_payload.get("model_specs", {})

    scores: Mapping[str, Any] = result_payload.get("scores", {})

    input_price = model_specs.get("input_price")
    output_price = model_specs.get("output_price")
    price = model_specs.get("price")

    # Derive a combined price when not explicitly provided. Many vendor
    # configs expose separate prompt (input) and completion (output) costs,
    # so we use their sum as an aggregate price column for reporting.
    if price is None:
        price_components = []
        if isinstance(input_price, (int, float)):
            price_components.append(float(input_price))
        if isinstance(output_price, (int, float)):
            price_components.append(float(output_price))
        if price_components:
            price = float(sum(price_components))

    return ReportRow(
        model_name=model_name,
        param_count=model_specs.get("param_count"),
        architecture=model_specs.get("architecture"),
        input_price=input_price,
        output_price=output_price,
        price=price,
        entity_score=scores.get("entity_score"),
        dev_score=scores.get("dev_score"),
        community_score=scores.get("community_score"),
        technical_score=scores.get("technical_score"),
        final_score=scores.get("final_score"),
    )


def _write_csv(
    path: Path,
    headers: Sequence[str],
    rows: Iterable[ReportRow],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def generate_csv_report(
    models: Optional[Sequence[str]] = None,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    output_path: Optional[Path | str] = None,
    headers: Sequence[str] = DEFAULT_HEADERS,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Path:
    """Generate a CSV report summarizing scoring results.

    Args:
        models: Optional sequence of model names to include. When omitted all
            ``*_results.json`` files found in ``results_dir`` are exported.
        results_dir: Directory containing JSON scoring results.
        models_dir: Directory containing model metadata JSON files.
        output_path: Explicit output location. When omitted a timestamped file
            is created inside ``results_dir / Reports``.
        headers: Column order for the generated CSV. Defaults to
            :data:`DEFAULT_HEADERS`.
        project_name: Name used in the default filename.

    Returns:
        Path to the generated CSV file.

    Raises:
        FileNotFoundError: When the results directory does not exist or no
            matching result files were discovered.
        ValueError: When the headers collection is empty.
    """

    if not headers:
        raise ValueError("CSV headers cannot be empty.")

    results_dir_path = Path(results_dir)
    models_dir_path = Path(models_dir)
    output_path = Path(output_path) if output_path else None

    result_files = _collect_result_files(results_dir_path, models=models)
    if not result_files:
        raise FileNotFoundError(
            f"No result files found in {results_dir_path}"
            + (f" for models {list(models)}" if models else "")
        )

    rows: List[ReportRow] = []
    for result_file in result_files:
        payload = _load_json(result_file)
        if not payload:
            LOGGER.warning("Skipping invalid result file: %s", result_file)
            continue
        rows.append(_build_row(result_file, payload, models_dir_path))

    if not rows:
        raise FileNotFoundError(
            "No valid result payloads were available for export."
        )

    if output_path is None:
        reports_directory = results_dir_path / DEFAULT_REPORTS_SUBDIR
        reports_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_name}_report_{timestamp}.csv"
        output_path = reports_directory / filename
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(output_path, headers, rows)
    LOGGER.info("CSV report generated at %s", output_path)
    return output_path


__all__ = ["generate_csv_report", "DEFAULT_HEADERS", "ReportRow"]
