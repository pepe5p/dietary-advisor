"""CLI smoke tests for the evaluation app."""

from __future__ import annotations

from typer.testing import CliRunner

from evaluation.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-cases" in result.stdout
    assert "evaluate" in result.stdout


def test_run_cases_help() -> None:
    result = runner.invoke(app, ["run-cases", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.stdout
    assert "--repeats" not in result.stdout


def test_evaluate_pending() -> None:
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 1
    assert "pending rework" in result.stdout
