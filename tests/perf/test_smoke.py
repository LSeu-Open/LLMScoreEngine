"""Performance harness benchmarks for core scoring/reporting actions."""
from __future__ import annotations

from itertools import islice
from pathlib import Path
from time import perf_counter

import pytest

from llmscore.actions.catalog import register_default_actions
from llmscore.actions.registry import ActionRegistry

pytestmark = pytest.mark.perf

SCORE_BATCH_MAX_SECONDS = 20.0
LEADERBOARD_MAX_SECONDS = 5.0


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    reg = ActionRegistry()
    register_default_actions(reg)
    return reg


@pytest.fixture(scope="module")
def sample_models() -> list[str]:
    models_dir = Path("Models")
    assert models_dir.exists(), "Models directory missing for perf tests"
    models = [path.stem for path in models_dir.glob("*.json")]
    assert models, "No model fixtures found for performance benchmarks"
    return models[: min(10, len(models))]


def test_score_batch_benchmark(benchmark, registry, sample_models, tmp_path):
    """Benchmark the score.batch action with a representative sample set."""

    inputs = {
        "models": sample_models,
        "models_dir": "Models",
        "results_dir": str(tmp_path / "results"),
        "quiet": True,
    }

    start = perf_counter()
    registry.run("score.batch", inputs=inputs)
    warmup_duration = perf_counter() - start
    assert warmup_duration < SCORE_BATCH_MAX_SECONDS, (
        f"score.batch exceeded budget: {warmup_duration:.2f}s"
    )

    def _run():
        return registry.run("score.batch", inputs=inputs)

    result = benchmark(_run)
    assert result.output["successes"] == len(sample_models)


def test_results_leaderboard_benchmark(benchmark, registry, tmp_path):
    """Benchmark results.leaderboard against an existing results directory."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Seed results dir by running a small batch first
    models_dir = Path("Models")
    sample = [path.stem for path in islice(models_dir.glob("*.json"), 3)]
    batch_inputs = {
        "models": sample,
        "models_dir": "Models",
        "results_dir": str(results_dir),
        "quiet": True,
    }
    registry.run("score.batch", inputs=batch_inputs)

    leaderboard_inputs = {"sort_key": "final_score", "limit": 10}

    start = perf_counter()
    registry.run("results.leaderboard", inputs=leaderboard_inputs)
    warmup_duration = perf_counter() - start
    assert warmup_duration < LEADERBOARD_MAX_SECONDS, (
        f"results.leaderboard exceeded budget: {warmup_duration:.2f}s"
    )

    def _run():
        return registry.run("results.leaderboard", inputs=leaderboard_inputs)

    result = benchmark(_run)
    assert "entries" in result.output
    assert len(result.output["entries"]) <= 10
