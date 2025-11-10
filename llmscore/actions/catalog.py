"""Default action catalog for llmscore."""

from __future__ import annotations

import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
import yaml

from config import scoring_config as default_scoring_config
from model_scoring.core.constants import MODELS_DIR, RESULTS_DIR
from model_scoring.run_scoring import run_scoring
from model_scoring.utils.config_loader import load_config_from_path
from model_scoring.utils.logging import configure_console_only_logging
from model_scoring.utils.csv_reporter import generate_csv_report

try:
    from model_scoring.utils.graph_reporter import (
        generate_report as generate_html_report,
    )
except Exception:  # pragma: no cover - optional dependency
    generate_html_report = None

from ..analytics import (
    BenchmarkAnalytics,
    DataAuthoring,
    DebugAnalytics,
    ModelAnalytics,
    ResultsAnalytics,
)
from ..automation.scheduler import (
    ScheduleJob,
    SchedulerService,
    WebhookConfig,
    WebhookNotifier,
)
from ..automation.watchers import WatchService, build_watch_job
from ..state.profiles import Profile, ProfileManager
from ..state.store import SessionRecord, SessionStore
from ..utils.logging import get_logger
from ..workflows.registry import WorkflowRegistry
from ..workflows.models import WorkflowDefinition, WorkflowStep
from .base import ActionDefinition, ActionMetadata
from .registry import ActionRegistry

LOGGER = get_logger("llmscore.actions")
PIPELINE_MODULE = None
SESSION_STORE: SessionStore | None = None
PROFILE_MANAGER: ProfileManager | None = None
RESULTS_ANALYTICS: ResultsAnalytics | None = None
MODELS_ANALYTICS: ModelAnalytics | None = None
BENCHMARK_ANALYTICS: BenchmarkAnalytics | None = None
DATA_AUTHORING: DataAuthoring | None = None
DEBUG_ANALYTICS: DebugAnalytics | None = None
PIPELINE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "fill-benchmark-pipeline"
    / "llm_benchmark_pipeline.py"
)

ACTIVE_AUTOMATION: Dict[str, Dict[str, Any]] = {}


def _get_pipeline_module():
    global PIPELINE_MODULE  # type: ignore[global-variable-not-assigned]
    if PIPELINE_MODULE is None:
        if not PIPELINE_MODULE_PATH.exists():
            raise FileNotFoundError(
                f"Pipeline script not found at {PIPELINE_MODULE_PATH}"
            )
        spec = importlib.util.spec_from_file_location(
            "llmscore.pipeline.fill_benchmark", PIPELINE_MODULE_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load benchmark pipeline module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        PIPELINE_MODULE = module  # type: ignore[assignment]
    return PIPELINE_MODULE


def _load_scoring_config(config_path: Optional[str]):
    if not config_path:
        return None
    return load_config_from_path(config_path)


def _write_result(
    output_dir: Optional[Path],
    model: str,
    payload: Dict[str, Any],
) -> Optional[Path]:
    if not output_dir:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model}_results.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return output_path


def _maybe_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value)


def _load_pipeline_primitives():
    module = _get_pipeline_module()
    return (
        getattr(module, "PipelineConfig"),
        getattr(module, "LLMBenchmarkPipeline"),
        getattr(module, "load_config_from_file"),
    )


def _get_session_store() -> SessionStore:
    global SESSION_STORE  # type: ignore[global-variable-not-assigned]
    if SESSION_STORE is None:
        SESSION_STORE = SessionStore()
    return SESSION_STORE


def _get_profile_manager() -> ProfileManager:
    global PROFILE_MANAGER  # type: ignore[global-variable-not-assigned]
    if PROFILE_MANAGER is None:
        PROFILE_MANAGER = ProfileManager(_get_session_store())
    return PROFILE_MANAGER


def _get_results_analytics() -> ResultsAnalytics:
    global RESULTS_ANALYTICS  # type: ignore[global-variable-not-assigned]
    if RESULTS_ANALYTICS is None:
        RESULTS_ANALYTICS = ResultsAnalytics(
            session_store=_get_session_store()
        )
    return RESULTS_ANALYTICS


def _get_models_analytics() -> ModelAnalytics:
    global MODELS_ANALYTICS  # type: ignore[global-variable-not-assigned]
    if MODELS_ANALYTICS is None:
        MODELS_ANALYTICS = ModelAnalytics()
    return MODELS_ANALYTICS


def _get_benchmark_analytics() -> BenchmarkAnalytics:
    global BENCHMARK_ANALYTICS  # type: ignore[global-variable-not-assigned]
    if BENCHMARK_ANALYTICS is None:
        BENCHMARK_ANALYTICS = BenchmarkAnalytics()
    return BENCHMARK_ANALYTICS


def _get_data_authoring() -> DataAuthoring:
    global DATA_AUTHORING  # type: ignore[global-variable-not-assigned]
    if DATA_AUTHORING is None:
        DATA_AUTHORING = DataAuthoring()
    return DATA_AUTHORING


def _get_debug_analytics() -> DebugAnalytics:
    global DEBUG_ANALYTICS  # type: ignore[global-variable-not-assigned]
    if DEBUG_ANALYTICS is None:
        DEBUG_ANALYTICS = DebugAnalytics()
    return DEBUG_ANALYTICS


def _build_pipeline_config(
    config_path: Optional[str],
    overrides: Dict[str, Any],
):
    PipelineConfig, _, load_config_from_file = _load_pipeline_primitives()
    config_data: Dict[str, Any] = {}
    if config_path:
        config_data.update(load_config_from_file(config_path))
    for key, value in overrides.items():
        if value is not None:
            config_data[key] = value
    return PipelineConfig(**config_data)


def _load_models_payload(
    models: Optional[Sequence[Any]],
    models_file: Optional[str],
) -> List[Dict[str, Any]]:
    entries: List[Any] = []
    if models_file:
        path = Path(models_file)
        if not path.exists():
            raise FileNotFoundError(f"Models file not found: {models_file}")
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            if "models" in loaded and isinstance(loaded["models"], list):
                entries.extend(loaded["models"])
            else:
                raise ValueError(
                    "Models file must contain a list or a dict with a "
                    "'models' key."
                )
        elif isinstance(loaded, list):
            entries.extend(loaded)
        else:
            raise ValueError("Models file must decode to a list or dict.")
    if models:
        entries.extend(models)
    if not entries:
        raise ValueError("No models provided.")

    normalized: List[Dict[str, Any]] = []
    for item in entries:
        if isinstance(item, str):
            normalized.append({"name": item})
        elif isinstance(item, dict):
            if "name" not in item:
                raise ValueError(
                    "Model dictionary entries must include a 'name'."
                )
            normalized.append(item)
        else:
            raise ValueError("Model entries must be strings or dictionaries.")
    return normalized


def _normalize_string_sequence(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(item) for item in decoded]
            return [str(decoded)]
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ValueError("Expected a string or list-like value.")


def _score_report_action_handler(**kwargs: Any) -> Dict[str, Any]:
    models = _normalize_string_sequence(kwargs.get("models"))
    results_dir = kwargs.get("results_dir") or RESULTS_DIR
    models_dir = kwargs.get("models_dir") or MODELS_DIR
    project_name = kwargs.get("project_name") or "LLM-Scoring-Engine"
    reports_dir = kwargs.get("output_dir")
    reports_path = (
        Path(reports_dir)
        if reports_dir
        else Path(results_dir) / "Reports"
    )
    reports_path.mkdir(parents=True, exist_ok=True)

    csv_enabled = kwargs.get("csv", True)
    html_enabled = kwargs.get("html", True)

    csv_report: Optional[Path] = None
    if csv_enabled:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_output = reports_path / f"{project_name}_report_{timestamp}.csv"
        csv_report = generate_csv_report(
            models=models,
            results_dir=results_dir,
            models_dir=models_dir,
            output_path=csv_output,
            project_name=project_name,
        )

    html_report: Optional[str] = None
    if html_enabled:
        if generate_html_report is None:
            raise RuntimeError(
                "HTML report generation is not available. Ensure the "
                "graph reporter dependencies are installed."
            )
        html_result = generate_html_report(
            template_dir=kwargs.get("template_dir")
        )
        html_report = (
            html_result
            if isinstance(html_result, str)
            else str(reports_path / "model_performance_report.html")
        )

    return {
        "reports": {
            "csv": str(csv_report) if csv_report else None,
            "html": html_report,
        }
    }


def _data_template_action_handler(**kwargs: Any) -> Dict[str, Any]:
    authoring = _get_data_authoring()
    return authoring.template(
        kwargs["model_name"],
        overwrite=kwargs.get("overwrite", False),
    )


def _data_validate_action_handler(**kwargs: Any) -> Dict[str, Any]:
    authoring = _get_data_authoring()
    models = _normalize_string_sequence(kwargs.get("models"))
    if not models:
        models = [
            path.stem
            for path in Path(MODELS_DIR).glob("*.json")
        ]
    return authoring.validate(models, strict=kwargs.get("strict", True))


def _models_list_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_models_analytics().list()


def _models_info_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_models_analytics().info(kwargs["model"])


def _models_search_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_models_analytics().search(
        kwargs["query"],
        limit=kwargs.get("limit", 5),
    )


def _models_tag_action_handler(**kwargs: Any) -> Dict[str, Any]:
    tags = _normalize_string_sequence(kwargs.get("tags")) or []
    return _get_models_analytics().tag(
        kwargs["model"],
        tags,
        replace=kwargs.get("replace", False),
    )


def _models_sync_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_models_analytics().sync(source=kwargs.get("source"))


def _benchmark_list_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_benchmark_analytics().list()


def _benchmark_info_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_benchmark_analytics().info(kwargs["name"])


def _benchmark_update_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_benchmark_analytics().update(
        kwargs["name"],
        kwargs["source"],
    )


def _benchmark_validate_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_benchmark_analytics().validate(kwargs["name"])


def _results_list_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_results_analytics().list(limit=kwargs.get("limit"))


def _results_show_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_results_analytics().show(kwargs["model"])


def _results_compare_action_handler(**kwargs: Any) -> Dict[str, Any]:
    metrics = _normalize_string_sequence(kwargs.get("metrics"))
    return _get_results_analytics().compare(
        kwargs["primary"],
        kwargs["secondary"],
        metrics=metrics,
    )


def _results_export_action_handler(**kwargs: Any) -> Dict[str, Any]:
    models = _normalize_string_sequence(kwargs.get("models")) or []
    if not models:
        raise ValueError("At least one model must be specified.")
    return _get_results_analytics().export(
        models,
        output_dir=kwargs["output_dir"],
        format=kwargs.get("format", "json"),
    )


def _results_leaderboard_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_results_analytics().leaderboard(
        sort_key=kwargs.get("sort_key", "final_score"),
        limit=kwargs.get("limit"),
    )


def _results_analyze_failures_action_handler(
    **kwargs: Any,
) -> Dict[str, Any]:
    return _get_results_analytics().analyze_failures(
        kwargs["model"],
        minimum_score=kwargs.get("minimum_score", 0.0),
    )


def _results_pin_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_results_analytics().pin(
        kwargs["model"],
        note=kwargs.get("note"),
        profile=kwargs.get("profile"),
    )


def _debug_inspect_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _get_debug_analytics().inspect(kwargs["model"])


def _data_fill_action_handler(**kwargs: Any) -> Dict[str, Any]:
    PipelineConfig, LLMBenchmarkPipeline, _ = _load_pipeline_primitives()
    overrides = {
        "template_path": kwargs.get("template_path"),
        "output_dir": kwargs.get("output_dir"),
        "artificial_analysis_key": kwargs.get("aa_key"),
        "huggingface_key": kwargs.get("hf_key"),
        "rate_limit_aa": kwargs.get("rate_limit_aa"),
        "rate_limit_hf": kwargs.get("rate_limit_hf"),
        "max_retries": kwargs.get("max_retries"),
        "retry_backoff_factor": kwargs.get("retry_backoff_factor"),
        "timeout": kwargs.get("timeout"),
        "continue_on_error": kwargs.get("continue_on_error"),
    }
    config = _build_pipeline_config(
        kwargs.get("config_path"),
        overrides,
    )
    pipeline = LLMBenchmarkPipeline(config)
    models = _load_models_payload(
        kwargs.get("models"),
        kwargs.get("models_file"),
    )
    results = pipeline.process_batch(models)
    successes = sum(1 for item in results if item.get("status") == "success")
    return {
        "config": (
            config.model_dump()
            if hasattr(config, "model_dump")
            else config.__dict__
        ),
        "models_processed": len(models),
        "successes": successes,
        "failures": len(results) - successes,
        "results": results,
        "output_dir": str(config.output_dir),
    }


def _session_save_action_handler(**kwargs: Any) -> Dict[str, Any]:
    store = _get_session_store()
    record = SessionRecord(
        identifier=kwargs["identifier"],
        data=kwargs.get("data") or {},
        profile=kwargs.get("profile"),
    )
    saved = store.save(record)
    return {
        "identifier": saved.identifier,
        "profile": saved.profile,
        "created_at": saved.created_at.isoformat()
        if saved.created_at
        else None,
        "updated_at": saved.updated_at.isoformat()
        if saved.updated_at
        else None,
    }


def _session_load_action_handler(**kwargs: Any) -> Dict[str, Any]:
    store = _get_session_store()
    record = store.load(kwargs["identifier"])
    if not record:
        raise ValueError(f"Session '{kwargs['identifier']}' not found.")
    return {
        "identifier": record.identifier,
        "profile": record.profile,
        "data": record.data,
        "created_at": record.created_at.isoformat()
        if record.created_at
        else None,
        "updated_at": record.updated_at.isoformat()
        if record.updated_at
        else None,
    }


def _session_list_action_handler(**kwargs: Any) -> Dict[str, Any]:
    store = _get_session_store()
    records = store.list(
        profile=kwargs.get("profile"),
        limit=kwargs.get("limit"),
    )
    payload = []
    for record in records:
        payload.append(
            {
                "identifier": record.identifier,
                "profile": record.profile,
                "created_at": record.created_at.isoformat()
                if record.created_at
                else None,
                "updated_at": record.updated_at.isoformat()
                if record.updated_at
                else None,
            }
        )
    return {"sessions": payload}


def _session_delete_action_handler(**kwargs: Any) -> Dict[str, Any]:
    store = _get_session_store()
    store.delete(kwargs["identifier"])
    return {"deleted": kwargs["identifier"]}


def _workspace_init_action_handler(**kwargs: Any) -> Dict[str, Any]:
    manager = _get_profile_manager()
    profile = Profile(
        name=kwargs["name"],
        workspace_path=kwargs["workspace_path"],
        default_session=kwargs.get("default_session"),
    )
    saved = manager.add(profile)
    return {
        "name": saved.name,
        "workspace_path": saved.workspace_path,
        "default_session": saved.default_session,
        "created_at": saved.created_at.isoformat()
        if saved.created_at
        else None,
        "updated_at": saved.updated_at.isoformat()
        if saved.updated_at
        else None,
    }


def _workspace_list_action_handler(**kwargs: Any) -> Dict[str, Any]:
    manager = _get_profile_manager()
    profiles = manager.list()
    payload = []
    for profile in profiles:
        payload.append(
            {
                "name": profile.name,
                "workspace_path": profile.workspace_path,
                "default_session": profile.default_session,
                "created_at": profile.created_at.isoformat()
                if profile.created_at
                else None,
                "updated_at": profile.updated_at.isoformat()
                if profile.updated_at
                else None,
            }
        )
    return {"profiles": payload}


def _workspace_set_default_session_handler(**kwargs: Any) -> Dict[str, Any]:
    manager = _get_profile_manager()
    profile = manager.set_default_session(
        kwargs["name"],
        kwargs.get("session_id"),
    )
    if not profile:
        raise ValueError(f"Workspace '{kwargs['name']}' not found.")
    return {
        "name": profile.name,
        "default_session": profile.default_session,
    }


def _score_single_model(
    model: str,
    *,
    models_dir: Optional[Path] = None,
    quiet: bool = False,
    config_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    scoring_config_module: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    configure_console_only_logging(quiet=quiet)
    scoring_config = (
        scoring_config_module
        or _load_scoring_config(config_path)
        or default_scoring_config
    )
    kwargs: Dict[str, Any] = {
        "quiet": quiet,
        "scoring_config": scoring_config,
    }
    if models_dir:
        kwargs["models_directory"] = str(models_dir)
    result = run_scoring(model, **kwargs)
    if result is None:
        raise RuntimeError(f"Scoring failed for model '{model}'.")

    output_path = _write_result(output_dir, model, result)
    payload: Dict[str, Any] = {
        "model": model,
        "scores": result.get("scores", {}),
        "output_path": str(output_path) if output_path else None,
    }
    return payload


def _score_batch(
    models: Sequence[str],
    *,
    models_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    quiet: bool = False,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    configure_console_only_logging(quiet=quiet)
    scoring_config = (
        _load_scoring_config(config_path) or default_scoring_config
    )
    start = datetime.utcnow()
    summary: List[Dict[str, Any]] = []
    for model in models:
        try:
            payload = _score_single_model(
                model,
                models_dir=models_dir,
                quiet=quiet,
                config_path=config_path,
                scoring_config_module=scoring_config,
                output_dir=results_dir,
            )
            payload["status"] = "success"
            summary.append(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to score model %s: %s", model, exc)
            summary.append(
                {
                    "model": model,
                    "status": "error",
                    "error": str(exc),
                }
            )
    elapsed = (datetime.utcnow() - start).total_seconds()
    successes = sum(1 for item in summary if item.get("status") == "success")
    return {
        "models": list(models),
        "successes": successes,
        "failures": len(summary) - successes,
        "elapsed_seconds": elapsed,
        "results": summary,
    }


def _score_model_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _score_single_model(
        kwargs["model"],
        models_dir=_maybe_path(kwargs.get("models_dir")),
        quiet=kwargs.get("quiet", False),
        config_path=kwargs.get("config_path"),
        output_dir=_maybe_path(kwargs.get("output_dir")),
    )


def _score_batch_action_handler(**kwargs: Any) -> Dict[str, Any]:
    return _score_batch(
        kwargs["models"],
        models_dir=_maybe_path(kwargs.get("models_dir")),
        results_dir=_maybe_path(kwargs.get("results_dir")),
        quiet=kwargs.get("quiet", False),
        config_path=kwargs.get("config_path"),
    )


def register_scoring_actions(registry: ActionRegistry) -> None:
    """Register scoring related actions into the registry."""

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="score.model",
                title="Score Model",
                description="Run scoring for a single model JSON.",
                domain="scoring",
                tags=("scoring", "model"),
            ),
            handler=_score_model_action_handler,
            input_schema={
                "model": {
                    "type": "string",
                    "description": (
                        "Model name (JSON filename sans extension)."
                    ),
                },
                "models_dir": {
                    "type": "string",
                    "description": "Directory containing model JSON files.",
                },
                "output_dir": {
                    "type": "string",
                    "description": (
                        "Directory to write scoring results JSON."
                    ),
                },
                "quiet": {"type": "boolean", "default": False},
                "config_path": {
                    "type": "string",
                    "description": (
                        "Override scoring configuration file path."
                    ),
                },
            },
            output_schema={
                "model": {"type": "string"},
                "scores": {"type": "object"},
                "output_path": {"type": ["string", "null"]},
            },
            examples=("score.model model=DeepSeek-R1",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="score.batch",
                title="Score Models Batch",
                description=(
                    "Run scoring for multiple models and aggregate results."
                ),
                domain="scoring",
                tags=("scoring", "batch"),
            ),
            handler=_score_batch_action_handler,
            input_schema={
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model names.",
                },
                "models_dir": {"type": "string"},
                "results_dir": {"type": "string"},
                "quiet": {"type": "boolean", "default": False},
                "config_path": {"type": "string"},
            },
            output_schema={
                "models": {"type": "array", "items": {"type": "string"}},
                "successes": {"type": "integer"},
                "failures": {"type": "integer"},
                "elapsed_seconds": {"type": "number"},
                "results": {"type": "array", "items": {"type": "object"}},
            },
            examples=(
                "score.batch models='[\"DeepSeek-R1\",\"GPT-4\"]'",
            ),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="score.report",
                title="Generate Score Reports",
                description=(
                    "Generate CSV and HTML reports summarizing scoring "
                    "results."
                ),
                domain="scoring",
                tags=("scoring", "report"),
            ),
            handler=_score_report_action_handler,
            input_schema={
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional subset of models to include in the report."
                    ),
                },
                "results_dir": {
                    "type": "string",
                    "description": "Path to the results directory.",
                },
                "models_dir": {
                    "type": "string",
                    "description": "Path to the models directory.",
                },
                "output_dir": {
                    "type": "string",
                    "description": (
                        "Directory where reports should be written. "
                        "Defaults to Results/Reports."
                    ),
                },
                "project_name": {
                    "type": "string",
                    "description": "Prefix used for generated filenames.",
                },
                "template_dir": {
                    "type": "string",
                    "description": "Override location for HTML templates.",
                },
                "csv": {"type": "boolean", "default": True},
                "html": {"type": "boolean", "default": True},
            },
            output_schema={
                "reports": {
                    "type": "object",
                    "properties": {
                        "csv": {"type": ["string", "null"]},
                        "html": {"type": ["string", "null"]},
                    },
                }
            },
            examples=(
                "score.report",
                "score.report models='[\"DeepSeek-R1\"]' csv=true html=false",
            ),
        ),
        replace=True,
    )


def register_data_actions(registry: ActionRegistry) -> None:
    """Register data management actions."""

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="data.fill",
                title="Fill Benchmark Data",
                description=(
                    "Fill model benchmark templates using the pipeline "
                    "with Artificial Analysis and HuggingFace sources."
                ),
                domain="data",
                tags=("data", "pipeline"),
            ),
            handler=_data_fill_action_handler,
            input_schema={
                "models": {
                    "type": "array",
                    "items": {"type": ["string", "object"]},
                    "description": (
                        "Optional inline model definitions. Each entry can "
                        "be a name string or a dictionary with metadata."
                    ),
                },
                "models_file": {
                    "type": "string",
                    "description": (
                        "Path to a JSON/YAML file containing a 'models' list "
                        "definition."
                    ),
                },
                "config_path": {
                    "type": "string",
                    "description": "Configuration file supplying defaults.",
                },
                "template_path": {"type": "string"},
                "output_dir": {"type": "string"},
                "aa_key": {"type": "string"},
                "hf_key": {"type": "string"},
                "rate_limit_aa": {"type": "number"},
                "rate_limit_hf": {"type": "number"},
                "max_retries": {"type": "integer"},
                "retry_backoff_factor": {"type": "number"},
                "timeout": {"type": "integer"},
                "continue_on_error": {"type": "boolean"},
            },
            output_schema={
                "config": {"type": "object"},
                "models_processed": {"type": "integer"},
                "successes": {"type": "integer"},
                "failures": {"type": "integer"},
                "results": {"type": "array", "items": {"type": "object"}},
                "output_dir": {"type": "string"},
            },
            examples=(
                "data.fill template_path=Templates/base.json "
                "models='[\"DeepSeek-R1\"]'",
            ),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="data.template",
                title="Generate Model Template",
                description=(
                    "Generate a starter JSON template for a model definition."
                ),
                domain="data",
                tags=("data", "template"),
            ),
            handler=_data_template_action_handler,
            input_schema={
                "model_name": {
                    "type": "string",
                    "description": "Name of the model template to create.",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Replace the template if it already exists."
                    ),
                },
            },
            output_schema={"path": {"type": "string"}},
            examples=("data.template model_name=MyModel",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="data.validate",
                title="Validate Model Data",
                description="Validate one or more model JSON files.",
                domain="data",
                tags=("data", "validate"),
            ),
            handler=_data_validate_action_handler,
            input_schema={
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names of models to validate. Defaults to all models "
                        "when omitted."
                    ),
                },
                "strict": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Stop on first validation failure when true."
                    ),
                },
            },
            output_schema={
                "results": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
            examples=("data.validate models='[\"DeepSeek-R1\"]'",),
        ),
        replace=True,
    )


def register_results_actions(registry: ActionRegistry) -> None:
    """Register analytics actions for scoring results."""

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="results.list",
                title="List Results",
                description="List stored scoring results ordered by recency.",
                domain="results",
                tags=("results", "analytics"),
            ),
            handler=_results_list_action_handler,
            input_schema={
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to return.",
                }
            },
            output_schema={
                "count": {"type": "integer"},
                "results": {"type": "array", "items": {"type": "object"}},
            },
            examples=("results.list limit=5",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="results.show",
                title="Show Result",
                description="Display detailed scoring output for a model.",
                domain="results",
                tags=("results", "analytics"),
            ),
            handler=_results_show_action_handler,
            input_schema={"model": {"type": "string"}},
            output_schema={"type": "object"},
            examples=("results.show model=DeepSeek-R1",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="results.compare",
                title="Compare Results",
                description="Compare two models across scoring metrics.",
                domain="results",
                tags=("results", "analytics"),
            ),
            handler=_results_compare_action_handler,
            input_schema={
                "primary": {"type": "string"},
                "secondary": {"type": "string"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional subset of metrics to compare. Defaults to "
                        "all available metrics."
                    ),
                },
            },
            output_schema={
                "primary": {"type": "string"},
                "secondary": {"type": "string"},
                "metrics": {"type": "array", "items": {"type": "object"}},
            },
            examples=(
                "results.compare primary=DeepSeek-R1 secondary=GPT-4",
            ),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="results.export",
                title="Export Results",
                description="Export results to JSON or CSV formats.",
                domain="results",
                tags=("results", "export"),
            ),
            handler=_results_export_action_handler,
            input_schema={
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Models to export.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Destination directory for exported files.",
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "csv"],
                    "default": "json",
                },
            },
            output_schema={
                "exported": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            examples=(
                "results.export models='[\"DeepSeek-R1\"]' "
                "output_dir=Exports format=json",
            ),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="results.leaderboard",
                title="Results Leaderboard",
                description=(
                    "Generate a leaderboard of models sorted by score."
                ),
                domain="results",
                tags=("results", "analytics"),
            ),
            handler=_results_leaderboard_action_handler,
            input_schema={
                "sort_key": {
                    "type": "string",
                    "default": "final_score",
                    "description": "Metric used for sorting.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of entries to include in the table."
                    ),
                },
            },
            output_schema={
                "entries": {"type": "array", "items": {"type": "object"}},
                "sort_key": {"type": "string"},
            },
            examples=("results.leaderboard limit=10",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="results.analyze_failures",
                title="Analyze Result Failures",
                description=(
                    "Surface metrics or entries that failed validation or "
                    "fell below a threshold."
                ),
                domain="results",
                tags=("results", "analytics"),
            ),
            handler=_results_analyze_failures_action_handler,
            input_schema={
                "model": {"type": "string"},
                "minimum_score": {
                    "type": "number",
                    "default": 0.0,
                    "description": (
                        "Highlight metrics below this score threshold."
                    ),
                },
            },
            output_schema={
                "model": {"type": "string"},
                "failures": {"type": "array", "items": {"type": "object"}},
            },
            examples=("results.analyze_failures model=DeepSeek-R1",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="results.pin",
                title="Pin Result",
                description="Pin a result for quick access in the session.",
                domain="results",
                tags=("results", "state"),
            ),
            handler=_results_pin_action_handler,
            input_schema={
                "model": {"type": "string"},
                "note": {"type": "string"},
                "profile": {"type": "string"},
            },
            output_schema={
                "identifier": {"type": "string"},
                "profile": {"type": ["string", "null"]},
                "path": {"type": ["string", "null"]},
                "pinned_at": {"type": ["string", "null"]},
            },
            examples=(
                "results.pin model=DeepSeek-R1 "
                "note='Release candidate'",
            ),
        ),
        replace=True,
    )


def register_models_actions(registry: ActionRegistry) -> None:
    """Register model intelligence actions."""

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="models.list",
                title="List Models",
                description="List available model definitions.",
                domain="models",
                tags=("models", "catalog"),
            ),
            handler=_models_list_action_handler,
            input_schema={},
            output_schema={
                "models": {"type": "array", "items": {"type": "object"}},
                "count": {"type": "integer"},
            },
            examples=("models.list",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="models.info",
                title="Model Info",
                description="Display detailed information for a model.",
                domain="models",
                tags=("models", "catalog"),
            ),
            handler=_models_info_action_handler,
            input_schema={"model": {"type": "string"}},
            output_schema={
                "name": {"type": "string"},
                "path": {"type": "string"},
                "data": {"type": "object"},
            },
            examples=("models.info model=DeepSeek-R1",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="models.search",
                title="Search Models",
                description="Search model definitions using fuzzy matching.",
                domain="models",
                tags=("models", "search"),
            ),
            handler=_models_search_action_handler,
            input_schema={
                "query": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum matches to return.",
                },
            },
            output_schema={
                "query": {"type": "string"},
                "matches": {"type": "array", "items": {"type": "object"}},
            },
            examples=("models.search query=deepseek limit=5",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="models.tag",
                title="Tag Model",
                description="Add or replace tags for a model.",
                domain="models",
                tags=("models", "metadata"),
            ),
            handler=_models_tag_action_handler,
            input_schema={
                "model": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to add to the model.",
                },
                "replace": {
                    "type": "boolean",
                    "default": False,
                    "description": "Replace existing tags when true.",
                },
            },
            output_schema={
                "model": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            examples=("models.tag model=DeepSeek-R1 tags='[\"beta\"]'",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="models.sync",
                title="Sync Models",
                description=(
                    "Placeholder for syncing model metadata with external "
                    "registries."
                ),
                domain="models",
                tags=("models", "sync"),
            ),
            handler=_models_sync_action_handler,
            input_schema={
                "source": {
                    "type": "string",
                    "description": "Optional source descriptor for logging.",
                }
            },
            output_schema={
                "status": {"type": "string"},
                "message": {"type": "string"},
                "source": {"type": ["string", "null"]},
            },
            examples=("models.sync source=hf",),
        ),
        replace=True,
    )


def register_benchmark_actions(registry: ActionRegistry) -> None:
    """Register benchmark catalogue actions."""

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="benchmark.list",
                title="List Benchmarks",
                description="List known benchmark collections.",
                domain="benchmark",
                tags=("benchmark", "catalog"),
            ),
            handler=_benchmark_list_action_handler,
            input_schema={},
            output_schema={
                "benchmarks": {"type": "array", "items": {"type": "object"}},
            },
            examples=("benchmark.list",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="benchmark.info",
                title="Benchmark Info",
                description="Display metadata for a specific benchmark.",
                domain="benchmark",
                tags=("benchmark", "catalog"),
            ),
            handler=_benchmark_info_action_handler,
            input_schema={"name": {"type": "string"}},
            output_schema={
                "name": {"type": "string"},
                "path": {"type": "string"},
                "metadata": {"type": "object"},
            },
            examples=("benchmark.info name=entity_benchmarks",),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="benchmark.update",
                title="Update Benchmark",
                description="Update or add a benchmark definition file.",
                domain="benchmark",
                tags=("benchmark", "maintenance"),
            ),
            handler=_benchmark_update_action_handler,
            input_schema={
                "name": {"type": "string"},
                "source": {
                    "type": "string",
                    "description": "Path to the benchmark definition file.",
                },
            },
            output_schema={
                "name": {"type": "string"},
                "path": {"type": "string"},
                "updated_at": {"type": "string"},
            },
            examples=(
                "benchmark.update name=custom "
                "source=benchmarks/custom.json",
            ),
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="benchmark.validate",
                title="Validate Benchmark",
                description="Validate the structure of a benchmark file.",
                domain="benchmark",
                tags=("benchmark", "validate"),
            ),
            handler=_benchmark_validate_action_handler,
            input_schema={"name": {"type": "string"}},
            output_schema={
                "name": {"type": "string"},
                "task_count": {"type": "integer"},
            },
            examples=("benchmark.validate name=custom",),
        ),
        replace=True,
    )


def register_debug_actions(registry: ActionRegistry) -> None:
    """Register debugging utilities."""

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="debug.inspect",
                title="Inspect Evaluation",
                description=(
                    "Inspect evaluation data for a model, including source "
                    "JSON and scoring results."
                ),
                domain="debug",
                tags=("debug", "inspection"),
            ),
            handler=_debug_inspect_action_handler,
            input_schema={"model": {"type": "string"}},
            output_schema={
                "result": {"type": "object"},
                "model": {"type": "object"},
            },
            examples=("debug.inspect model=DeepSeek-R1",),
        ),
        replace=True,
    )


def register_state_actions(registry: ActionRegistry) -> None:
    """Register session and workspace state actions."""

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="session.save",
                title="Save Session",
                description="Persist session data to the session store.",
                domain="session",
                tags=("session", "state"),
            ),
            handler=_session_save_action_handler,
            input_schema={
                "identifier": {"type": "string"},
                "data": {"type": "object"},
                "profile": {"type": ["string", "null"]},
            },
            output_schema={
                "identifier": {"type": "string"},
                "profile": {"type": ["string", "null"]},
                "created_at": {"type": ["string", "null"]},
                "updated_at": {"type": ["string", "null"]},
            },
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="session.load",
                title="Load Session",
                description="Load a session record by identifier.",
                domain="session",
                tags=("session", "state"),
            ),
            handler=_session_load_action_handler,
            input_schema={"identifier": {"type": "string"}},
            output_schema={
                "identifier": {"type": "string"},
                "profile": {"type": ["string", "null"]},
                "data": {"type": "object"},
                "created_at": {"type": ["string", "null"]},
                "updated_at": {"type": ["string", "null"]},
            },
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="session.list",
                title="List Sessions",
                description="List saved sessions filtered by profile.",
                domain="session",
                tags=("session", "state"),
            ),
            handler=_session_list_action_handler,
            input_schema={
                "profile": {"type": ["string", "null"]},
                "limit": {"type": ["integer", "null"]},
            },
            output_schema={
                "sessions": {"type": "array", "items": {"type": "object"}},
            },
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="session.delete",
                title="Delete Session",
                description="Remove a session record from the store.",
                domain="session",
                tags=("session", "state"),
            ),
            handler=_session_delete_action_handler,
            input_schema={"identifier": {"type": "string"}},
            output_schema={"deleted": {"type": "string"}},
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workspace.init",
                title="Initialize Workspace",
                description="Create a workspace profile entry.",
                domain="workspace",
                tags=("workspace", "state"),
            ),
            handler=_workspace_init_action_handler,
            input_schema={
                "name": {"type": "string"},
                "workspace_path": {"type": "string"},
                "default_session": {"type": ["string", "null"]},
            },
            output_schema={
                "name": {"type": "string"},
                "workspace_path": {"type": "string"},
                "default_session": {"type": ["string", "null"]},
                "created_at": {"type": ["string", "null"]},
                "updated_at": {"type": ["string", "null"]},
            },
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workspace.list",
                title="List Workspaces",
                description="List registered workspace profiles.",
                domain="workspace",
                tags=("workspace", "state"),
            ),
            handler=_workspace_list_action_handler,
            input_schema={},
            output_schema={
                "profiles": {"type": "array", "items": {"type": "object"}},
            },
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workspace.set_default_session",
                title="Set Workspace Default Session",
                description="Set the default session for a workspace profile.",
                domain="workspace",
                tags=("workspace", "state"),
            ),
            handler=_workspace_set_default_session_handler,
            input_schema={
                "name": {"type": "string"},
                "session_id": {"type": ["string", "null"]},
            },
            output_schema={
                "name": {"type": "string"},
                "default_session": {"type": ["string", "null"]},
            },
        ),
        replace=True,
    )


def register_workflow_actions(registry: ActionRegistry) -> None:
    """Register workflow marketplace and management actions."""

    def _registry(profile: Optional[str]) -> WorkflowRegistry:
        return _get_workflow_registry(profile)

    def list_handler(**kwargs: Any) -> Dict[str, Any]:
        profile = kwargs.get("profile")
        workflows = _registry(profile).list()
        return {
            "workflows": [
                {
                    "name": item.name,
                    "description": item.description,
                    "version": item.version,
                    "tags": list(item.tags),
                    "author": item.author,
                    "source": item.source,
                }
                for item in workflows
            ]
        }

    def show_handler(**kwargs: Any) -> Dict[str, Any]:
        profile = kwargs.get("profile")
        workflow = _registry(profile).get(kwargs["name"])
        if not workflow:
            raise ValueError(f"Workflow '{kwargs['name']}' not found")
        return {"workflow": workflow.to_dict()}

    def import_handler(**kwargs: Any) -> Dict[str, Any]:
        profile = kwargs.get("profile")
        overwrite = _coerce_bool(kwargs.get("overwrite"), default=False)
        source = kwargs.get("source")
        registry_obj = _registry(profile)
        if kwargs.get("url"):
            payload = _load_remote_workflow(kwargs["url"])
            definition = WorkflowDefinition.from_dict(payload)
            definition.source = source or kwargs["url"]
        elif kwargs.get("path"):
            path = Path(kwargs["path"])
            if not path.exists():
                raise FileNotFoundError(f"Workflow file not found: {path}")
            text = path.read_text(encoding="utf-8")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = yaml.safe_load(text)
            if not isinstance(payload, dict):
                raise ValueError("Workflow document must decode to an object.")
            definition = WorkflowDefinition.from_dict(payload)
            definition.source = source or str(path)
        else:
            raise ValueError("Either 'path' or 'url' must be provided.")
        _validate_workflow_actions(registry, definition)
        saved = registry_obj.save(definition, overwrite=overwrite)
        return {"workflow": saved.to_dict()}

    def export_handler(**kwargs: Any) -> Dict[str, Any]:
        profile = kwargs.get("profile")
        path = Path(kwargs["path"])
        registry_obj = _registry(profile)
        registry_obj.export_to_path(kwargs["name"], path)
        return {"path": str(path)}

    def delete_handler(**kwargs: Any) -> Dict[str, Any]:
        profile = kwargs.get("profile")
        registry_obj = _registry(profile)
        registry_obj.delete(kwargs["name"])
        return {"deleted": kwargs["name"]}

    def create_handler(**kwargs: Any) -> Dict[str, Any]:
        profile = kwargs.get("profile")
        tags = _normalize_string_sequence(kwargs.get("tags")) or []
        steps_payload = kwargs.get("steps")
        if steps_payload is None:
            raise ValueError("Workflow steps are required.")
        steps = _coerce_steps_payload(steps_payload)
        definition = WorkflowDefinition(
            name=kwargs["name"],
            description=kwargs.get("description", ""),
            steps=steps,
            version=str(kwargs.get("version", "1.0")),
            tags=tuple(tags),
            author=kwargs.get("author"),
            source=kwargs.get("source"),
        )
        _validate_workflow_actions(registry, definition)
        saved = _registry(profile).save(
            definition,
            overwrite=_coerce_bool(kwargs.get("overwrite"), default=False),
        )
        return {"workflow": saved.to_dict()}

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workflow.list",
                title="List Workflows",
                description="List saved workflow definitions.",
                domain="workflow",
                tags=("workflow", "list"),
            ),
            handler=list_handler,
            input_schema={"profile": {"type": ["string", "null"]}},
            output_schema={
                "workflows": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workflow.show",
                title="Show Workflow",
                description="Display a workflow definition by name.",
                domain="workflow",
                tags=("workflow", "show"),
            ),
            handler=show_handler,
            input_schema={
                "name": {"type": "string"},
                "profile": {"type": ["string", "null"]},
            },
            output_schema={"workflow": {"type": "object"}},
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workflow.import",
                title="Import Workflow",
                description="Import a workflow definition from path or URL.",
                domain="workflow",
                tags=("workflow", "import"),
            ),
            handler=import_handler,
            input_schema={
                "path": {"type": ["string", "null"]},
                "url": {"type": ["string", "null"]},
                "profile": {"type": ["string", "null"]},
                "overwrite": {"type": ["boolean", "null"]},
                "source": {"type": ["string", "null"]},
            },
            output_schema={"workflow": {"type": "object"}},
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workflow.export",
                title="Export Workflow",
                description="Export a workflow definition to disk.",
                domain="workflow",
                tags=("workflow", "export"),
            ),
            handler=export_handler,
            input_schema={
                "name": {"type": "string"},
                "path": {"type": "string"},
                "profile": {"type": ["string", "null"]},
            },
            output_schema={"path": {"type": "string"}},
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workflow.delete",
                title="Delete Workflow",
                description="Remove a workflow definition.",
                domain="workflow",
                tags=("workflow", "delete"),
            ),
            handler=delete_handler,
            input_schema={
                "name": {"type": "string"},
                "profile": {"type": ["string", "null"]},
            },
            output_schema={"deleted": {"type": "string"}},
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="workflow.create",
                title="Create Workflow",
                description="Create a workflow definition from step payload.",
                domain="workflow",
                tags=("workflow", "create"),
            ),
            handler=create_handler,
            input_schema={
                "name": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "object"}},
                "description": {"type": ["string", "null"]},
                "profile": {"type": ["string", "null"]},
                "overwrite": {"type": ["boolean", "null"]},
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "version": {"type": ["string", "null"]},
                "author": {"type": ["string", "null"]},
                "source": {"type": ["string", "null"]},
            },
            output_schema={"workflow": {"type": "object"}},
        ),
        replace=True,
    )


def register_automation_actions(registry: ActionRegistry) -> None:
    """Register automation helpers for watch and schedule flows."""

    def _runner(action_name: str, inputs: Dict[str, Any]) -> None:
        registry.run(action_name, inputs=inputs)

    def watch_handler(**kwargs: Any) -> Dict[str, Any]:
        paths = _normalize_string_sequence(kwargs.get("paths"))
        if not paths:
            raise ValueError("At least one path must be provided.")
        job = build_watch_job(
            paths,
            action=kwargs["action"],
            inputs=_coerce_object(kwargs.get("inputs")),
            recursive=_coerce_bool(kwargs.get("recursive"), default=True),
            patterns=_normalize_string_sequence(kwargs.get("patterns")),
            ignore_patterns=_normalize_string_sequence(
                kwargs.get("ignore_patterns")
            ),
            debounce_seconds=float(kwargs.get("debounce_seconds", 1.0)),
        )
        notifier = _build_webhook_notifier(kwargs)
        notify_callback = notifier.send if notifier else None
        service = WatchService(job, _runner, notify=notify_callback)
        watch_id = kwargs.get("identifier") or f"watch::{uuid4().hex}"
        paths_resolved = [
            str(target.normalized_path()) for target in job.targets
        ]
        background = _coerce_bool(kwargs.get("background"), default=False)
        if background:
            service.start()
            ACTIVE_AUTOMATION[watch_id] = {
                "type": "watch",
                "service": service,
                "job": job,
                "action": job.action,
                "paths": paths_resolved,
            }
            return {
                "identifier": watch_id,
                "status": "watching",
                "mode": "background",
                "paths": paths_resolved,
            }
        try:
            service.run_forever()
        finally:
            service.stop()
        return {
            "identifier": watch_id,
            "status": "stopped",
            "mode": "foreground",
            "paths": paths_resolved,
        }

    def schedule_handler(**kwargs: Any) -> Dict[str, Any]:
        tz_value = kwargs.get("timezone")
        job = ScheduleJob(
            action=kwargs["action"],
            cron=kwargs["cron"],
            inputs=_coerce_object(kwargs.get("inputs")),
            tz=_resolve_timezone(tz_value),
        )
        notifier = _build_webhook_notifier(kwargs)
        service = SchedulerService(job, _runner, notifier=notifier)
        schedule_id = kwargs.get("identifier") or f"schedule::{uuid4().hex}"
        background = _coerce_bool(kwargs.get("background"), default=True)
        if background:
            service.start()
            ACTIVE_AUTOMATION[schedule_id] = {
                "type": "schedule",
                "service": service,
                "job": job,
                "action": job.action,
                "cron": job.cron,
            }
            return {
                "identifier": schedule_id,
                "status": "scheduled",
                "mode": "background",
                "cron": job.cron,
            }
        duration_value = kwargs.get("duration_seconds")
        service.start()
        try:
            if duration_value is not None:
                time.sleep(float(duration_value))
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:  # pragma: no cover - interactive use
            LOGGER.info("Scheduler interrupted by user")
        finally:
            service.stop()
        return {
            "identifier": schedule_id,
            "status": "stopped",
            "mode": "foreground",
            "cron": job.cron,
        }

    def automation_list_handler(**kwargs: Any) -> Dict[str, Any]:
        jobs = []
        for identifier, entry in ACTIVE_AUTOMATION.items():
            payload = {
                "identifier": identifier,
                "type": entry.get("type"),
                "action": entry.get("action"),
            }
            if entry.get("type") == "watch":
                payload["paths"] = entry.get("paths")
            if entry.get("type") == "schedule":
                payload["cron"] = entry.get("cron")
            jobs.append(payload)
        return {"automation": jobs}

    def automation_stop_handler(**kwargs: Any) -> Dict[str, Any]:
        identifier = kwargs["identifier"]
        entry = ACTIVE_AUTOMATION.pop(identifier, None)
        if not entry:
            raise ValueError(f"Automation job '{identifier}' not found")
        service = entry["service"]
        service.stop()
        return {"identifier": identifier, "status": "stopped"}

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="automation.watch",
                title="Watch Directory",
                description="Watch directories and trigger actions on change.",
                domain="automation",
                tags=("automation", "watch"),
            ),
            handler=watch_handler,
            input_schema={
                "paths": {"type": "array", "items": {"type": "string"}},
                "action": {"type": "string"},
                "inputs": {"type": ["object", "null"]},
                "recursive": {"type": ["boolean", "null"]},
                "patterns": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "ignore_patterns": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "debounce_seconds": {"type": ["number", "null"]},
                "background": {"type": ["boolean", "null"]},
                "identifier": {"type": ["string", "null"]},
                "webhook_url": {"type": ["string", "null"]},
                "webhook_method": {"type": ["string", "null"]},
                "webhook_headers": {"type": ["object", "null"]},
                "webhook_payload_key": {"type": ["string", "null"]},
                "webhook_timeout": {"type": ["number", "null"]},
            },
            output_schema={"identifier": {"type": "string"}},
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="automation.schedule",
                title="Schedule Action",
                description="Schedule actions on a cron expression.",
                domain="automation",
                tags=("automation", "schedule"),
            ),
            handler=schedule_handler,
            input_schema={
                "action": {"type": "string"},
                "cron": {"type": "string"},
                "inputs": {"type": ["object", "null"]},
                "timezone": {"type": ["string", "null"]},
                "background": {"type": ["boolean", "null"]},
                "duration_seconds": {"type": ["number", "null"]},
                "identifier": {"type": ["string", "null"]},
                "webhook_url": {"type": ["string", "null"]},
                "webhook_method": {"type": ["string", "null"]},
                "webhook_headers": {"type": ["object", "null"]},
                "webhook_payload_key": {"type": ["string", "null"]},
                "webhook_timeout": {"type": ["number", "null"]},
            },
            output_schema={"identifier": {"type": "string"}},
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="automation.list",
                title="List Automation Jobs",
                description="List active automation jobs.",
                domain="automation",
                tags=("automation", "list"),
            ),
            handler=automation_list_handler,
            input_schema={},
            output_schema={
                "automation": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
        ),
        replace=True,
    )

    registry.register(
        ActionDefinition(
            metadata=ActionMetadata(
                name="automation.stop",
                title="Stop Automation Job",
                description="Stop a background automation job by identifier.",
                domain="automation",
                tags=("automation", "stop"),
            ),
            handler=automation_stop_handler,
            input_schema={"identifier": {"type": "string"}},
            output_schema={"identifier": {"type": "string"}},
        ),
        replace=True,
    )


def register_default_actions(registry: ActionRegistry) -> None:
    """Register all default actions."""

    register_scoring_actions(registry)
    register_data_actions(registry)
    register_results_actions(registry)
    register_models_actions(registry)
    register_benchmark_actions(registry)
    register_debug_actions(registry)
    register_state_actions(registry)
    register_workflow_actions(registry)
    register_automation_actions(registry)


__all__ = [
    "register_default_actions",
    "register_scoring_actions",
    "register_data_actions",
    "register_results_actions",
    "register_models_actions",
    "register_benchmark_actions",
    "register_debug_actions",
    "register_state_actions",
    "register_workflow_actions",
    "register_automation_actions",
]


def _coerce_object(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid json
            raise ValueError("Expected JSON object string.") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Expected JSON object string.")
        return dict(decoded)
    raise ValueError("Expected mapping or JSON string.")


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError("Expected boolean or boolean-like string.")


def _coerce_steps_payload(value: Any) -> List[WorkflowStep]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid json
            raise ValueError("Workflow steps must be JSON array.") from exc
    else:
        decoded = value
    if not isinstance(decoded, Sequence):
        raise ValueError("Workflow steps must be provided as a list.")
    steps: List[WorkflowStep] = []
    for item in decoded:
        if not isinstance(item, dict):
            raise ValueError("Workflow steps must be dictionaries.")
        if "action" not in item:
            raise ValueError("Workflow step dictionaries require an 'action'.")
        steps.append(WorkflowStep.from_dict(item))
    return steps


def _resolve_timezone(name: Optional[str]) -> timezone:
    if not name:
        return timezone.utc
    if name.lower() == "utc":
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unknown timezone '{name}'.") from exc


def _build_webhook_notifier(
    params: Dict[str, Any],
) -> Optional[WebhookNotifier]:
    url = params.get("webhook_url")
    if not url:
        return None
    headers = _coerce_object(params.get("webhook_headers"))
    timeout_value = params.get("webhook_timeout", 10.0)
    timeout = float(timeout_value) if timeout_value is not None else 10.0
    config = WebhookConfig(
        url=str(url),
        method=str(params.get("webhook_method", "POST")),
        headers={str(key): str(value) for key, value in headers.items()},
        payload_key=str(params.get("webhook_payload_key", "message")),
        timeout_seconds=timeout,
    )
    return WebhookNotifier(config)


def _get_workflow_registry(profile: Optional[str] = None) -> WorkflowRegistry:
    return WorkflowRegistry(_get_session_store(), profile=profile)


def _validate_workflow_actions(
    registry: ActionRegistry,
    definition: WorkflowDefinition,
) -> None:
    registered = set(registry.names())
    missing = [
        action
        for action in definition.required_actions()
        if action not in registered
    ]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Workflow references unknown actions: {joined}")


def _load_remote_workflow(url: str) -> Dict[str, Any]:
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()
    text = response.text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("Remote workflow must decode to an object.")
        return loaded
