"""CLI smoke tests for the evaluation app."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from evaluation.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-cases" in result.stdout
    assert "evaluate" in result.stdout
    assert "plot" in result.stdout


def test_run_cases_help() -> None:
    result = runner.invoke(app, ["run-cases", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.stdout
    assert "--repeats" not in result.stdout


def test_evaluate_help() -> None:
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.stdout
    assert "--force" in result.stdout


def test_plot_help() -> None:
    result = runner.invoke(app, ["plot", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.stdout
    assert "experiments" in result.stdout.lower()


def test_plot_exits_one_when_no_scored_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evaluation.cli._plot_experiment", lambda experiment: False)
    monkeypatch.setattr("evaluation.cli._plot_run_variability", lambda: False)
    result = runner.invoke(app, ["plot"])
    assert result.exit_code == 1
