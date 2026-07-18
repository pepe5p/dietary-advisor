"""CLI smoke tests using Typer's CliRunner."""

from __future__ import annotations

from typer.testing import CliRunner

from dietary_advisor.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "recommend" in result.stdout


def test_cli_info() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "llm_model" in result.stdout


def test_cli_recommend_unknown_profile_id() -> None:
    result = runner.invoke(app, ["recommend", "--profile-id", "does_not_exist"])
    assert result.exit_code != 0


def test_cli_recommend_requires_profile_or_profile_id() -> None:
    result = runner.invoke(app, ["recommend"])
    assert result.exit_code != 0
    assert "Pass exactly one of --profile or --profile-id" in result.output


def test_cli_recommend_rejects_both_profile_and_profile_id() -> None:
    result = runner.invoke(
        app,
        ["recommend", "--profile-id", "L1_01", "--profile", '{"user_id": "x"}'],
    )
    assert result.exit_code != 0
    assert "Pass exactly one of --profile or --profile-id" in result.output


def test_cli_recommend_rejects_invalid_profile_json() -> None:
    result = runner.invoke(app, ["recommend", "--profile", "not-json"])
    assert result.exit_code != 0
    assert "Invalid --profile JSON" in result.output
