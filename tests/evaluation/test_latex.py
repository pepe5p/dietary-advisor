"""Tests for LaTeX table generation."""

from __future__ import annotations

from evaluation.judges import all_judges
from evaluation.latex import model_summary_table, variant_summary_table
from tests.evaluation.conftest import make_scored_run

JUDGE_A, JUDGE_B = all_judges()[0], all_judges()[1]


def test_variant_summary_table_has_one_soft_column_per_judge() -> None:
    records_by_judge = {
        JUDGE_A.key: [make_scored_run(variant="baseline", soft=0.8)],
        JUDGE_B.key: [make_scored_run(variant="baseline", soft=0.6)],
    }
    table = variant_summary_table(records_by_judge)

    for judge in all_judges():
        assert f"Soft {judge.label}" in table
    assert "0.800" in table
    assert "0.600" in table


def test_variant_summary_table_has_one_safety_column_per_judge() -> None:
    records_by_judge = {
        JUDGE_A.key: [make_scored_run(variant="baseline", safety=1.0)],
        JUDGE_B.key: [make_scored_run(variant="baseline", safety=0.5)],
    }
    table = variant_summary_table(records_by_judge)

    for judge in all_judges():
        assert f"Safety {judge.label}" in table
    assert "1.000" in table
    assert "0.500" in table


def test_variant_summary_table_shows_dash_for_unscored_judge() -> None:
    table = variant_summary_table({JUDGE_A.key: [make_scored_run(variant="baseline", soft=0.8)]})

    assert f"Soft {JUDGE_B.label}" in table
    assert "--" in table


def test_model_summary_table_has_one_soft_column_per_judge() -> None:
    records_by_judge = {
        JUDGE_A.key: [make_scored_run(soft=0.75)],
        JUDGE_B.key: [make_scored_run(soft=0.55)],
    }
    table = model_summary_table(records_by_judge)

    for judge in all_judges():
        assert f"Soft {judge.label}" in table
    assert "0.750" in table
    assert "0.550" in table


def test_both_summary_tables_share_a_column_layout() -> None:
    records_by_judge = {judge.key: [make_scored_run()] for judge in all_judges()}

    def header(table: str) -> str:
        return table.splitlines()[2].split("&", 1)[1]

    assert header(variant_summary_table(records_by_judge)) == header(model_summary_table(records_by_judge))


def test_summary_tables_include_totaller_calls_mean() -> None:
    records_by_judge = {judge.key: [make_scored_run(variant="totaller", totaller_calls=2)] for judge in all_judges()}
    table = variant_summary_table(records_by_judge)

    assert "Calls" in table.splitlines()[2]
    assert "2.00" in table.splitlines()[4]


def test_variant_summary_table_uses_display_labels() -> None:
    records_by_judge = {
        JUDGE_A.key: [make_scored_run(variant="totaller+reflective-loop", soft=0.8)],
        JUDGE_B.key: [make_scored_run(variant="totaller+reflective-loop", soft=0.6)],
    }
    table = variant_summary_table(records_by_judge)

    assert "full" in table.splitlines()[4]
    assert "totaller+reflective-loop" not in table
