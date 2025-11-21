"""Service layer powering analytics-oriented actions."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from rapidfuzz import fuzz, process

from model_scoring.core.constants import (
    MODELS_DIR,
    REQUIRED_SECTIONS,
    RESULTS_DIR,
)
from model_scoring.data.loaders import find_model_file, load_json_file
from model_scoring.data.validators import validate_model_data
from model_scoring.utils.csv_reporter import generate_csv_report

from ..state.store import SessionRecord, SessionStore
from ..utils.logging import get_logger

LOGGER = get_logger("llmscore.analytics")


def _read_json(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        LOGGER.debug("JSON not found: %s", path)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Invalid JSON in %s: %s", path, exc)
    except OSError as exc:
        LOGGER.warning("Error reading %s: %s", path, exc)
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _stat_timestamp(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _default_results_dir(path: Optional[Path | str]) -> Path:
    if path:
        return Path(path)
    return Path(RESULTS_DIR)


def _default_models_dir(path: Optional[Path | str]) -> Path:
    if path:
        return Path(path)
    return Path(MODELS_DIR)


def _result_path_for(model_name: str, results_dir: Path) -> Path:
    filename = f"{model_name}_results.json"
    path = results_dir / filename
    if path.exists():
        return path
    candidates = list(results_dir.glob(f"*{model_name}*_results.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Result file not found for '{model_name}'.")


@dataclass(slots=True)
class ResultSummary:
    """Compact representation of a stored scoring result."""

    model: str
    final_score: Optional[float]
    path: Path
    updated_at: datetime

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "final_score": self.final_score,
            "path": str(self.path),
            "updated_at": self.updated_at.isoformat(),
        }


class ResultsAnalytics:
    """Operations for inspecting and exporting scoring results."""

    def __init__(
        self,
        *,
        results_dir: Optional[Path | str] = None,
        session_store: Optional[SessionStore] = None,
    ) -> None:
        self.results_dir = _default_results_dir(results_dir)
        self.session_store = session_store

    def _iter_result_files(self) -> Iterable[Path]:
        if not self.results_dir.exists():
            raise FileNotFoundError(
                f"Results directory not found: {self.results_dir}"
            )
        return sorted(
            self.results_dir.glob("*_results.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

    def list(self, *, limit: Optional[int] = None) -> Dict[str, Any]:
        summaries: List[ResultSummary] = []
        for path in self._iter_result_files():
            payload = _read_json(path)
            if not payload:
                continue
            scores = payload.get("scores", {})
            summary = ResultSummary(
                model=payload.get(
                    "model_name", path.stem.replace("_results", "")
                ),
                final_score=scores.get("final_score"),
                path=path,
                updated_at=_stat_timestamp(path),
            )
            summaries.append(summary)
            if limit and len(summaries) >= limit:
                break
        return {
            "count": len(summaries),
            "results": [item.as_dict() for item in summaries],
        }

    def show(self, model: str) -> Dict[str, Any]:
        path = _result_path_for(model, self.results_dir)
        payload = _read_json(path)
        if not payload:
            raise FileNotFoundError(
                f"Unable to load result payload for '{model}'."
            )
        payload.setdefault("path", str(path))
        payload.setdefault("updated_at", _stat_timestamp(path).isoformat())
        return payload  # type: ignore[return-value]

    def compare(
        self,
        primary: str,
        secondary: str,
        *,
        metrics: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        first = self.show(primary)
        second = self.show(secondary)

        scores_one = first.get("scores", {})
        scores_two = second.get("scores", {})
        metrics = metrics or sorted(
            {
                *scores_one.keys(),
                *scores_two.keys(),
            }
        )

        diffs: List[Dict[str, Any]] = []
        for metric in metrics:
            value_one = scores_one.get(metric)
            value_two = scores_two.get(metric)
            diff = (
                None
                if value_one is None or value_two is None
                else value_one - value_two
            )
            diffs.append(
                {
                    "metric": metric,
                    "primary": value_one,
                    "secondary": value_two,
                    "delta": diff,
                }
            )

        return {
            "primary": primary,
            "secondary": secondary,
            "metrics": diffs,
        }

    def export(
        self,
        models: Sequence[str],
        *,
        output_dir: Path | str,
        format: str = "json",
    ) -> Dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        exported: List[str] = []
        if format == "json":
            for model in models:
                src = _result_path_for(model, self.results_dir)
                dst = destination / src.name
                shutil.copy2(src, dst)
                exported.append(str(dst))
        elif format == "csv":
            report_path = destination / "results_report.csv"
            generate_csv_report(
                models=models,
                results_dir=self.results_dir,
                output_path=report_path,
            )
            exported.append(str(report_path))
        else:
            raise ValueError(
                "Unsupported export format. Choose from {'json', 'csv'}."
            )

        return {"exported": exported}

    def leaderboard(
        self,
        *,
        sort_key: str = "final_score",
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        for path in self._iter_result_files():
            payload = _read_json(path) or {}
            scores = payload.get("scores", {})
            entries.append(
                {
                    "model": payload.get(
                        "model_name", path.stem.replace("_results", "")
                    ),
                    "scores": scores,
                    "path": str(path),
                }
            )

        def _sort_key(item: Dict[str, Any]) -> float:
            score = item["scores"].get(sort_key)
            return float(score) if score is not None else float("-inf")

        entries.sort(key=_sort_key, reverse=True)
        if limit:
            entries = entries[:limit]
        return {"entries": entries, "sort_key": sort_key}

    def analyze_failures(
        self,
        model: str,
        *,
        minimum_score: float = 0.0,
    ) -> Dict[str, Any]:
        payload = self.show(model)
        failures: List[Dict[str, Any]] = []

        details = payload.get("failures") or []
        if isinstance(details, list) and details:
            failures.extend(details)  # type: ignore[arg-type]

        scores = payload.get("scores", {})
        for metric, value in scores.items():
            if value is None:
                failures.append(
                    {"metric": metric, "reason": "missing_score"}
                )
            elif value < minimum_score:
                failures.append(
                    {
                        "metric": metric,
                        "reason": "below_threshold",
                        "value": value,
                        "threshold": minimum_score,
                    }
                )

        return {"model": model, "failures": failures}

    def pin(
        self,
        model: str,
        *,
        note: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.session_store:
            self.session_store = SessionStore()

        payload = self.show(model)
        record = SessionRecord(
            identifier=f"results.pin::{model}",
            profile=profile,
            category="pinned_context",
            data={
                "model": model,
                "note": note,
                "path": payload.get("path"),
                "pinned_at": datetime.now(UTC).isoformat(),
            },
        )
        saved = self.session_store.save(record)
        return {
            "identifier": saved.identifier,
            "profile": saved.profile,
            "path": saved.data.get("path"),
            "pinned_at": saved.data.get("pinned_at"),
        }


class ModelAnalytics:
    """Metadata utilities for model JSON definitions."""

    def __init__(self, *, models_dir: Optional[Path | str] = None) -> None:
        self.models_dir = _default_models_dir(models_dir)

    def _iter_models(self) -> Iterable[Path]:
        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Models directory not found: {self.models_dir}"
            )
        return sorted(self.models_dir.glob("*.json"))

    def _load_model(self, model_name: str) -> Dict[str, Any]:
        path_str = find_model_file(model_name, str(self.models_dir))
        if not path_str:
            raise FileNotFoundError(f"Model '{model_name}' not found.")
        payload = load_json_file(path_str)
        if payload is None:
            raise ValueError(f"Unable to load model definition: {model_name}")
        return payload

    def list(self) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        for path in self._iter_models():
            payload = _read_json(path) or {}
            specs = payload.get("model_specs", {})
            entries.append(
                {
                    "name": path.stem,
                    "architecture": specs.get("architecture"),
                    "param_count": specs.get("param_count"),
                    "tags": payload.get("tags", []),
                    "updated_at": _stat_timestamp(path).isoformat(),
                }
            )
        return {"models": entries, "count": len(entries)}

    def info(self, model_name: str) -> Dict[str, Any]:
        payload = self._load_model(model_name)
        path_str = find_model_file(model_name, str(self.models_dir))
        return {
            "name": model_name,
            "path": path_str,
            "data": payload,
        }

    def search(self, query: str, *, limit: int = 5) -> Dict[str, Any]:
        candidates = [path.stem for path in self._iter_models()]
        results = process.extract(
            query, candidates, scorer=fuzz.WRatio, limit=limit
        )
        matches = [
            {"name": name, "score": score, "index": idx}
            for name, score, idx in results
        ]
        return {"query": query, "matches": matches}

    def tag(
        self,
        model_name: str,
        tags: Sequence[str],
        *,
        replace: bool = False,
    ) -> Dict[str, Any]:
        path_str = find_model_file(model_name, str(self.models_dir))
        if not path_str:
            raise FileNotFoundError(f"Model '{model_name}' not found.")
        path = Path(path_str)
        payload = _read_json(path) or {}
        existing = list(payload.get("tags", []))
        if replace:
            combined = list(dict.fromkeys(tags))
        else:
            combined = list(dict.fromkeys([*existing, *tags]))
        payload["tags"] = combined
        _write_json(path, payload)
        return {"model": model_name, "tags": combined}

    def sync(self, *, source: Optional[str] = None) -> Dict[str, Any]:
        message = (
            "Model registry synchronisation is not yet implemented. "
            "Use manual updates or provide a future adapter."
        )
        payload: Dict[str, Any] = {"status": "pending", "message": message}
        if source:
            payload["source"] = source
        return payload


class BenchmarkAnalytics:
    """Helpers for benchmark catalogue management."""

    def __init__(self, *, benchmarks_dir: Optional[Path | str] = None) -> None:
        self.benchmarks_dir = (
            Path(benchmarks_dir) if benchmarks_dir else Path("Benchmarks")
        )
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

    def _benchmark_path(self, name: str) -> Path:
        path = self.benchmarks_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark '{name}' not found.")
        return path

    def list(self) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        for path in sorted(self.benchmarks_dir.glob("*.json")):
            payload = _read_json(path) or {}
            entries.append(
                {
                    "name": path.stem,
                    "tasks": sorted(payload.get("tasks", [])),
                    "updated_at": _stat_timestamp(path).isoformat(),
                }
            )
        if not entries:
            entries = [
                {
                    "name": section,
                    "tasks": sorted(REQUIRED_SECTIONS.get(section, [])),
                    "updated_at": None,
                }
                for section in ("entity_benchmarks", "dev_benchmarks")
            ]
        return {"benchmarks": entries}

    def info(self, name: str) -> Dict[str, Any]:
        path = self._benchmark_path(name)
        payload = _read_json(path) or {}
        return {
            "name": name,
            "path": str(path),
            "metadata": payload,
        }

    def update(self, name: str, source: str) -> Dict[str, Any]:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source benchmark not found: {source}")
        destination = self.benchmarks_dir / f"{name}{source_path.suffix}"
        shutil.copy2(source_path, destination)
        return {
            "name": name,
            "path": str(destination),
            "updated_at": _stat_timestamp(destination).isoformat(),
        }

    def validate(self, name: str) -> Dict[str, Any]:
        path = self._benchmark_path(name)
        payload = _read_json(path)
        if payload is None:
            raise ValueError(f"Could not parse benchmark '{name}'.")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("Benchmark payload must define a 'tasks' list.")
        return {"name": name, "task_count": len(tasks)}


class DataAuthoring:
    """Utilities for authoring and validating model JSON templates."""

    def __init__(self, *, models_dir: Optional[Path | str] = None) -> None:
        self.models_dir = _default_models_dir(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def template(
        self,
        model_name: str,
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        path = self.models_dir / f"{model_name}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Model template already exists: {path}"
            )

        payload: Dict[str, Any] = {
            "model_name": model_name,
            "model_specs": {
                field: None for field in REQUIRED_SECTIONS["model_specs"]
            },
            "entity_benchmarks": {
                field: None
                for field in REQUIRED_SECTIONS["entity_benchmarks"]
            },
            "dev_benchmarks": {
                field: None for field in REQUIRED_SECTIONS["dev_benchmarks"]
            },
            "community_score": {
                field: None for field in REQUIRED_SECTIONS["community_score"]
            },
            "tags": [],
        }
        _write_json(path, payload)
        return {"path": str(path)}

    def validate(
        self,
        models: Sequence[str],
        *,
        strict: bool = True,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for name in models:
            try:
                payload = self._load_model_payload(name)
                validate_model_data(payload, name)
                results.append({"model": name, "status": "valid"})
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {"model": name, "status": "invalid", "error": str(exc)}
                )
                if strict:
                    break
        return {"results": results}

    def _load_model_payload(self, model_name: str) -> Dict[str, Any]:
        path_str = find_model_file(model_name, str(self.models_dir))
        if not path_str:
            raise FileNotFoundError(f"Model '{model_name}' not found.")
        payload = load_json_file(path_str)
        if payload is None:
            raise ValueError(f"Unable to load JSON for '{model_name}'.")
        return payload


class DebugAnalytics:
    """Debug helpers for inspecting stored evaluations."""

    def __init__(
        self,
        *,
        results_dir: Optional[Path | str] = None,
        models_dir: Optional[Path | str] = None,
    ) -> None:
        self.results_dir = _default_results_dir(results_dir)
        self.models_dir = _default_models_dir(models_dir)

    def inspect(self, model: str) -> Dict[str, Any]:
        results = ResultsAnalytics(results_dir=self.results_dir).show(model)
        model_data = ModelAnalytics(models_dir=self.models_dir).info(model)

        return {
            "result": results,
            "model": model_data,
        }


__all__ = [
    "BenchmarkAnalytics",
    "DataAuthoring",
    "DebugAnalytics",
    "ModelAnalytics",
    "ResultsAnalytics",
]
