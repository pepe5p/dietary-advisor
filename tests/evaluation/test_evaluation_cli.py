"""CLI smoke tests for the evaluation ablation app."""

from __future__ import annotations

from typer.testing import CliRunner

from evaluation.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--repeats" in result.stdout
