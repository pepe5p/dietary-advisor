"""CLI smoke tests using Typer's CliRunner."""

from __future__ import annotations

from typer.testing import CliRunner

from dietary_advisor.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PROFILE_ID" in result.stdout


def test_cli_unknown_profile_id() -> None:
    result = runner.invoke(app, ["does_not_exist"])
    assert result.exit_code == 2
    assert "Unknown profile id" in result.output


def test_cli_requires_profile_id() -> None:
    result = runner.invoke(app, ["--query", "one day"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output
    assert "PROFILE_ID" in result.output
