"""CLI exec mode smoke tests ensuring registry wiring is intact."""
from __future__ import annotations

from itertools import islice

import pytest

from llmscore.actions.catalog import register_default_actions
from llmscore.actions.registry import ActionRegistry
from llmscore.__main__ import app


@pytest.fixture(scope="module")
def action_names() -> list[str]:
    registry = ActionRegistry()
    register_default_actions(registry)
    return list(registry.names())


def test_exec_help_lists_options(cli_runner):
    result = cli_runner.invoke(app, ["exec", "--help"])
    assert result.exit_code == 0
    assert "Execute a single action" in result.output


def test_exec_fails_for_unknown_action(cli_runner):
    result = cli_runner.invoke(app, ["exec", "does.not.exist"])
    assert result.exit_code != 0
    assert "Unknown action" in result.output


def test_exec_each_registered_action_has_help(cli_runner, action_names):
    for name in action_names:
        result = cli_runner.invoke(app, ["exec", name, "--help"])
        assert result.exit_code == 0, result.output


def test_exec_runs_sample_of_registered_actions(cli_runner, stub_registry, action_names):
    sampled = list(islice(action_names, 5))
    for name in sampled:
        result = cli_runner.invoke(app, ["exec", name])
        assert result.exit_code == 0, result.output
    assert len(stub_registry.calls) == len(sampled)


def test_exec_propagates_action_errors(cli_runner, stub_registry):
    stub_registry.set_failure("score.model", RuntimeError("boom"))

    result = cli_runner.invoke(app, ["exec", "score.model"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "boom" in str(result.exception)


def test_exec_automation_stop_reports_missing_job(cli_runner, stub_registry):
    stub_registry.set_available_actions(["automation.stop"])
    stub_registry.set_failure("automation.stop", ValueError("job not found"))

    result = cli_runner.invoke(
        app,
        ["exec", "automation.stop", "identifier=watch::ghost"],
    )

    assert result.exit_code != 0
    assert "job not found" in result.output.lower()


def test_exec_automation_list_failure_bubbles(cli_runner, stub_registry):
    stub_registry.set_available_actions(["automation.list"])
    stub_registry.set_failure("automation.list", RuntimeError("list blew up"))

    result = cli_runner.invoke(app, ["exec", "automation.list"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "list blew up" in str(result.exception)
