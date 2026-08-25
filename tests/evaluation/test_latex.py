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
        assert rf"Soft agg.\ {judge.label}" in table
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

    assert rf"Soft agg.\ {JUDGE_B.label}" in table
    assert "--" in table


def test_model_summary_table_has_one_soft_column_per_judge() -> None:
    records_by_judge = {
        JUDGE_A.key: [make_scored_run(soft=0.75)],
        JUDGE_B.key: [make_scored_run(soft=0.55)],
    }
    table = model_summary_table(records_by_judge)

    for judge in all_judges():
        assert rf"Soft agg.\ {judge.label}" in table
    assert "0.750" in table
    assert "0.550" in table
