"""Tests for LaTeX table generation."""

from __future__ import annotations

from evaluation.judges import all_judges
from evaluation.latex import model_summary_table, paired_sign_test_table, variant_summary_table
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

    def metric_header(table: str) -> str:
        cells = [cell.strip() for cell in table.splitlines()[2].split("&")]
        mae = next(i for i, cell in enumerate(cells) if "MAE" in cell)
        return " & ".join(cells[mae:])

    assert metric_header(variant_summary_table(records_by_judge)) == metric_header(
        model_summary_table(records_by_judge)
    )
    assert "Effort" in model_summary_table(records_by_judge).splitlines()[2]


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


def test_paired_sign_test_table_has_p_column_and_comparison_rows() -> None:
    variants = ("baseline", "totaller", "reflective-loop", "totaller+reflective-loop")
    records = [
        make_scored_run(variant=variant, scenario_id=scenario, mae_pct=10.0 if variant == "baseline" else 4.0)
        for variant in variants
        for scenario in ("regular", "vegetarian-allergic")
    ]
    records_by_judge = {judge.key: records for judge in all_judges()}
    table = paired_sign_test_table(records_by_judge)

    header = table.splitlines()[2]
    assert "Comparison" in header
    assert "$p$" in header
    assert "Improved" in header
    comparisons = (
        r"\texttt{baseline} $\to$ \texttt{totaller-only}",
        r"\texttt{baseline} $\to$ \texttt{reflection-only}",
        r"\texttt{totaller-only} $\to$ \texttt{full}",
        r"\texttt{reflection-only} $\to$ \texttt{full}",
    )
    metric_labels = ["MAE"] + [f"{head} {judge.label}" for judge in all_judges() for head in ("Soft", "Safety")]
    for comparison in comparisons:
        rows = [line for line in table.splitlines() if line.startswith(comparison)]
        assert [row.split(" & ")[1] for row in rows] == metric_labels
    assert "S[table-format=1.3]" in table
    assert "0." in table
