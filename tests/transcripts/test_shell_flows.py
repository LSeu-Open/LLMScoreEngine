"""Golden transcript tests for CLI-assisted flows."""
from __future__ import annotations

from pathlib import Path

from llmscore.__main__ import app

from .recorder import TranscriptEvent, TranscriptRecorder

BASELINES_DIR = Path(__file__).with_suffix("").parent / "baselines"


def _record_event(cli_runner, argv: list[str]) -> TranscriptEvent:
    prompt = "llmscore " + " ".join(argv)
    result = cli_runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    return TranscriptEvent(prompt=prompt, response=result.output.strip())


def _assert_transcript(flow_name: str, events: list[TranscriptEvent]) -> None:
    TranscriptRecorder(BASELINES_DIR).assert_match(flow_name, events)


def test_workflow_run_transcript(
    cli_runner,
    stub_registry,
    stub_workflow_registry,
    sample_workflow_definition,
):
    stub_registry.set_available_actions(
        step.action for step in sample_workflow_definition.steps
    )

    event = _record_event(
        cli_runner,
        ["run", sample_workflow_definition.name],
    )

    _assert_transcript("nightly_score_workflow", [event])


def test_score_model_exec_transcript(cli_runner, stub_registry):
    stub_registry.set_available_actions(["score.model"])
    stub_registry.set_output(
        "score.model",
        {
            "status": "ok",
            "action": "score.model",
            "inputs": {"model": "alpha"},
            "scores": {"final_score": 0.9123},
        },
    )

    event = _record_event(
        cli_runner,
        ["exec", "score.model", "model=alpha"],
    )

    _assert_transcript("score_model_exec", [event])


def test_results_leaderboard_exec_transcript(cli_runner, stub_registry):
    stub_registry.set_available_actions(["results.leaderboard"])
    stub_registry.set_output(
        "results.leaderboard",
        {
            "entries": [
                {"model": "alpha", "scores": {"final_score": 0.95}},
                {"model": "beta", "scores": {"final_score": 0.90}},
            ],
            "sort_key": "final_score",
        },
    )

    event = _record_event(
        cli_runner,
        ["exec", "results.leaderboard", "limit=2"],
    )

    _assert_transcript("results_leaderboard_exec", [event])


def test_automation_list_exec_transcript(cli_runner, stub_registry):
    stub_registry.set_available_actions(["automation.list"])
    stub_registry.set_output(
        "automation.list",
        {
            "jobs": [
                {
                    "id": "watch-models",
                    "action": "score.batch",
                    "paths": ["models/"],
                }
            ]
        },
    )

    event = _record_event(cli_runner, ["exec", "automation.list"])

    _assert_transcript("automation_list_exec", [event])


def test_onboarding_flow_transcript(cli_runner, stub_registry):
    actions = ["data.template", "data.fill", "score.model"]
    stub_registry.set_available_actions(actions)
    stub_registry.set_output(
        "data.template",
        {
            "model": "gamma",
            "path": "models/gamma.json",
            "status": "created",
        },
    )
    stub_registry.set_output(
        "data.fill",
        {
            "model": "gamma",
            "status": "queued",
            "tasks": 4,
        },
    )
    stub_registry.set_output(
        "score.model",
        {
            "status": "ok",
            "action": "score.model",
            "inputs": {"model": "gamma"},
            "scores": {"final_score": 0.887},
        },
    )

    events = [
        _record_event(cli_runner, ["exec", "data.template", "model_name=gamma"]),
        _record_event(cli_runner, ["exec", "data.fill", "model=gamma"]),
        _record_event(cli_runner, ["exec", "score.model", "model=gamma"]),
    ]

    _assert_transcript("onboarding_flow", events)


def test_analytics_flow_transcript(cli_runner, stub_registry):
    actions = ["results.leaderboard", "results.compare"]
    stub_registry.set_available_actions(actions)
    stub_registry.set_output(
        "results.leaderboard",
        {
            "entries": [
                {"model": "sigma", "scores": {"final_score": 0.97}},
                {"model": "tau", "scores": {"final_score": 0.92}},
            ],
            "sort_key": "final_score",
        },
    )
    stub_registry.set_output(
        "results.compare",
        {
            "primary": "sigma",
            "secondary": "tau",
            "metrics": [
                {"metric": "final_score", "primary": 0.97, "secondary": 0.92, "delta": 0.05}
            ],
        },
    )

    events = [
        _record_event(cli_runner, ["exec", "results.leaderboard", "limit=5"]),
        _record_event(
            cli_runner,
            ["exec", "results.compare", "primary=sigma", "secondary=tau"],
        ),
    ]

    _assert_transcript("analytics_flow", events)


def test_automation_flow_transcript(cli_runner, stub_registry):
    actions = ["automation.watch", "automation.list"]
    stub_registry.set_available_actions(actions)
    stub_registry.set_output(
        "automation.watch",
        {
            "job_id": "watch-models",
            "action": "score.batch",
            "paths": ["models/"],
        },
    )
    stub_registry.set_output(
        "automation.list",
        {
            "jobs": [
                {
                    "id": "watch-models",
                    "action": "score.batch",
                    "paths": ["models/"],
                }
            ]
        },
    )

    events = [
        _record_event(
            cli_runner,
            [
                "exec",
                "automation.watch",
                "action=score.batch",
                "paths=['models/']",
            ],
        ),
        _record_event(cli_runner, ["exec", "automation.list"]),
    ]

    _assert_transcript("automation_flow", events)
