"""Tests for LaTeX table generation."""

from __future__ import annotations

from evaluation.judges import all_judges
from evaluation.latex import model_summary_table, variant_summary_table
from evaluation.scoring.store import ScoreRecord
from evaluation.validation.qualitative import QualitativeResult

JUDGE_A, JUDGE_B = all_judges()[0], all_judges()[1]


def _score_record(
    *,
    llm_model: str = "openrouter:openai/gpt-5.6-luna",
    variant: str = "baseline",
    soft: float = 0.8,
) -> ScoreRecord:
    return ScoreRecord(
        llm_model=llm_model,
        variant=variant,
        scenario_id="regular",
        judge_model="test-judge",
        mae_pct=10.0,
        mse_pct=100.0,
        per_nutrient_pct={"energy_kcal": 10.0},
        qualitative=QualitativeResult(aggregate=soft, safety_adherence=1.0),
        iterations=1,
        elapsed_s=2.0,
    )


def test_variant_summary_table_has_one_soft_column_per_judge() -> None:
    records_by_judge = {
        JUDGE_A.key: [_score_record(variant="baseline", soft=0.8)],
        JUDGE_B.key: [_score_record(variant="baseline", soft=0.6)],
    }
    table = variant_summary_table(records_by_judge)

    for judge in all_judges():
        assert rf"Soft agg.\ {judge.label}" in table
    assert "0.800" in table
    assert "0.600" in table


def test_variant_summary_table_shows_dash_for_unscored_judge() -> None:
    table = variant_summary_table({JUDGE_A.key: [_score_record(variant="baseline", soft=0.8)]})

    assert rf"Soft agg.\ {JUDGE_B.label}" in table
    assert "--" in table


def test_model_summary_table_has_one_soft_column_per_judge() -> None:
    records_by_judge = {
        JUDGE_A.key: [_score_record(soft=0.75)],
        JUDGE_B.key: [_score_record(soft=0.55)],
    }
    table = model_summary_table(records_by_judge)

    for judge in all_judges():
        assert rf"Soft agg.\ {judge.label}" in table
    assert "0.750" in table
    assert "0.550" in table
