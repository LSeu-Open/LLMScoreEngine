"""CLI run mode tests validating workflow execution wiring."""
from __future__ import annotations

import json

from llmscore.__main__ import app


def test_run_requires_workflow_name(cli_runner):
    result = cli_runner.invoke(app, ["run"])

    assert result.exit_code != 0
    assert "Workflow name is required" in result.output


def test_run_executes_all_workflow_steps(cli_runner, stub_registry, stub_workflow_registry, sample_workflow_definition):
    stub_registry.set_available_actions(step.action for step in sample_workflow_definition.steps)
    result = cli_runner.invoke(app, ["run", sample_workflow_definition.name])

    assert result.exit_code == 0, result.output
    assert len(stub_registry.calls) == len(sample_workflow_definition.steps)
    payload = json.loads(result.output)
    assert payload["workflow"] == sample_workflow_definition.name


def test_run_reports_missing_workflow(cli_runner, stub_workflow_registry):
    stub_workflow_registry.set_definition(None)

    result = cli_runner.invoke(app, ["run", "unknown"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_run_rejects_watch_and_cron_combinations(cli_runner):
    result = cli_runner.invoke(
        app,
        [
            "run",
            "nightly",
            "--watch",
            "tests",
            "--cron",
            "0 2 * * *",
        ],
    )

    assert result.exit_code != 0
    assert "Cannot use --watch and --cron" in result.output


def test_run_watch_requires_action(cli_runner):
    result = cli_runner.invoke(
        app,
        [
            "run",
            "--watch",
            "models",
        ],
    )

    assert result.exit_code != 0
    assert "--action is required" in result.output


def test_run_watch_background_emits_progress(cli_runner, stub_registry):
    stub_registry.set_output(
        "automation.watch",
        {
            "identifier": "watch::abc123",
            "status": "watching",
            "mode": "background",
            "progress": [
                {"percent": 10, "description": "warming up"},
                {"percent": 55, "description": "indexing"},
            ],
        },
    )

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--watch",
            "models",
            "--action",
            "score.batch",
            "--pattern",
            "*.json",
            "--ignore-pattern",
            "tmp/*",
            "--debounce",
            "2.5",
            "--background",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "watching"
    assert payload["progress"][0]["percent"] == 10
    assert len(stub_registry.calls) == 1
    call = stub_registry.calls[0]
    assert call["action"] == "automation.watch"
    watch_inputs = call["inputs"]
    assert watch_inputs["paths"] == ["models"]
    assert watch_inputs["action"] == "score.batch"
    assert watch_inputs["patterns"] == ["*.json"]
    assert watch_inputs["ignore_patterns"] == ["tmp/*"]
    assert watch_inputs["debounce_seconds"] == 2.5
    assert watch_inputs["background"] is True
