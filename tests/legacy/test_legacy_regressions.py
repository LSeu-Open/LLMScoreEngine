"""Automated regression harness for legacy scripts (Phase 4)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Iterable

import pytest

from score_models import main as score_models_main
from model_scoring.scoring import hf_score
from llmscore.actions.base import ActionExecutionResult, ActionMetadata

_PIPELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "fill-benchmark-pipeline"
    / "llm_benchmark_pipeline.py"
)
_PIPELINE_MODULE: ModuleType | None = None


def _load_pipeline_module() -> ModuleType:
    global _PIPELINE_MODULE  # type: ignore[global-variable-not-assigned]
    if _PIPELINE_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "legacy_benchmark_pipeline",
            _PIPELINE_PATH,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PIPELINE_MODULE = module
    return _PIPELINE_MODULE


def _load_pipeline_module_with_stubs(monkeypatch) -> ModuleType:
    if "requests" not in sys.modules:
        stub = types.ModuleType("requests")

        class _StubResponse:  # pragma: no cover - simple data holder
            status_code = 200

            def raise_for_status(self):  # noqa: D401 - test stub
                return None

            def json(self):
                return {}

        class _StubRequestException(RuntimeError):
            pass

        stub.Response = _StubResponse  # type: ignore[attr-defined]
        stub.exceptions = types.SimpleNamespace(  # type: ignore[attr-defined]
            RequestException=_StubRequestException,
            Timeout=_StubRequestException,
            ConnectionError=_StubRequestException,
            HTTPError=_StubRequestException,
        )

        def _raise(*_args, **_kwargs):  # pragma: no cover - defensive
            raise RuntimeError("requests access is stubbed in tests")

        stub.get = _raise  # type: ignore[attr-defined]
        stub.post = _raise  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "requests", stub)
    return _load_pipeline_module()


@pytest.fixture(autouse=True)
def _stub_action_registry(monkeypatch):
    class _StubActionRegistry:
        def __init__(self):
            self.calls = []

        def run(self, name: str, *, inputs=None, controller=None):
            inputs = inputs or {}
            models = list(inputs.get("models", []))
            results_dir = Path(inputs.get("results_dir", "Results"))
            results_dir.mkdir(parents=True, exist_ok=True)
            artifacts = []
            for model in models:
                payload = {
                    "model_name": model,
                    "scores": {"final_score": 0.42},
                }
                (results_dir / f"{model}_results.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                artifacts.append(
                    {
                        "model": model,
                        "status": "success",
                        "scores": payload["scores"],
                    }
                )
            output = {
                "results": artifacts,
                "successes": len(models),
                "failures": 0,
            }
            metadata = ActionMetadata(
                name=name,
                title=name,
                description="stub",
                domain="legacy",
            )
            return ActionExecutionResult(metadata=metadata, output=output)

    monkeypatch.setattr("score_models.ActionRegistry", _StubActionRegistry)
    monkeypatch.setattr("score_models.register_default_actions", lambda _registry: None)


def _write_models(tmp_path: Path, names: Iterable[str]) -> list[str]:
    models_dir = tmp_path / "Models"
    models_dir.mkdir()
    written = []
    for name in names:
        payload = {
            "entity_benchmarks": {
                "MMLU": 0.5,
                "GPQA": 0.4,
                "artificial_analysis": 0.3,
            },
            "dev_benchmarks": {"LiveCodeBench": 0.1, "AIME": 0.2},
            "community_score": {
                "lm_sys_arena_score": 1000,
                "hf_score": 7,
            },
            "model_specs": {
                "price": 1.0,
                "context_window": 8192,
                "param_count": 7,
                "architecture": "transformer",
            },
        }
        file_path = models_dir / f"{name}.json"
        file_path.write_text(json.dumps(payload), encoding="utf-8")
        written.append(name)
    return written

def test_score_models_legacy_cli_all(tmp_path, monkeypatch):
    models = _write_models(tmp_path, ["Alpha", "Beta"])
    results_dir = tmp_path / "Results"
    results_dir.mkdir()

    monkeypatch.chdir(tmp_path)

    score_models_main(["--all"])

    outputs = sorted(results_dir.glob("*_results.json"))
    assert len(outputs) == len(models)
    for path in outputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model_name"] in models
        assert payload["scores"]["final_score"] >= 0


@pytest.mark.parametrize("argv", [["SampleModel"], ["SampleModel", "--quiet"]])
def test_score_models_cli_single_model(tmp_path, monkeypatch, argv):
    _write_models(tmp_path, ["SampleModel"])
    monkeypatch.chdir(tmp_path)

    score_models_main(argv)

    output = tmp_path / "Results" / "SampleModel_results.json"
    assert output.exists()


def test_hf_score_cli_uses_mocked_metrics(monkeypatch):
    fake = type("FakeModel", (), {})()
    fake.downloads = 2048
    fake.likes = 256
    fake.created_at = hf_score.datetime.now(hf_score.timezone.utc)

    def _fake_model_info(name: str):  # pragma: no cover - simple stub
        assert name == "org/model"
        return fake

    monkeypatch.setattr(hf_score, "model_info", _fake_model_info)
    info = hf_score.main(["org/model"])
    assert info["model_name"] == "org/model"
    assert info["downloads in last 30 days"] == 2048
    assert info["total likes"] == 256
    assert info["community_score"] >= 0


def test_benchmark_pipeline_cli_dry_run(monkeypatch, tmp_path):
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "entity_benchmarks": {"MMLU": None},
                "dev_benchmarks": {"LiveCodeBench": None},
                "community_score": {"hf_score": None},
                "model_specs": {"price": None},
            }
        ),
        encoding="utf-8",
    )

    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps([{"name": "Demo"}]), encoding="utf-8")
    output_dir = tmp_path / "filled"
    output_dir.mkdir()

    pipeline_module = _load_pipeline_module_with_stubs(monkeypatch)

    class _StubPipeline:
        def __init__(self, config):  # pragma: no cover - simple struct
            self.config = config

        def process_batch(self, models):
            assert models == [{"name": "Demo"}]
            return [{"model": "Demo", "status": "success", "output": "demo.json"}]

    monkeypatch.setattr(pipeline_module, "LLMBenchmarkPipeline", _StubPipeline)
    monkeypatch.setattr(pipeline_module, "load_models", lambda path: json.loads(Path(path).read_text()))
    exit_code = pipeline_module.main(
        [
            "--template",
            str(template),
            "--models",
            str(models_file),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
