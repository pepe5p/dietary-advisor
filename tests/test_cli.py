"""CLI smoke tests using Typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dietary_advisor.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "recommend" in result.stdout
    assert "evaluate" in result.stdout


def test_cli_info() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "llm_model" in result.stdout


def test_cli_profile_lifecycle(tmp_path: Path) -> None:
    p = {
        "user_id": "cli_t",
        "age": 30,
        "sex": "male",
        "height_cm": 180,
        "weight_kg": 78,
        "allergens": [],
        "conditions": ["none"],
        "diet_pattern": "omnivore",
    }
    profile_file = tmp_path / "p.json"
    profile_file.write_text(json.dumps(p), encoding="utf-8")

    add = runner.invoke(app, ["profile", "add", str(profile_file)])
    assert add.exit_code == 0, add.stdout

    listed = runner.invoke(app, ["profile", "list"])
    assert listed.exit_code == 0
    assert "cli_t" in listed.stdout

    show = runner.invoke(app, ["profile", "show", "cli_t"])
    assert show.exit_code == 0
    assert '"user_id"' in show.stdout

    delete = runner.invoke(app, ["profile", "delete", "cli_t"])
    assert delete.exit_code == 0
    assert "Deleted" in delete.stdout


def test_cli_recommend_missing_profile() -> None:
    result = runner.invoke(app, ["recommend", "does_not_exist"])
    assert result.exit_code != 0
