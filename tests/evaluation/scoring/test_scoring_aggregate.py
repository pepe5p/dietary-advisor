"""Tests for variant-level score aggregation."""

from __future__ import annotations

from evaluation.scoring.aggregate import summarize, VariantSummary
from evaluation.scoring.store import ScoreRecord
from evaluation.validation.qualitative import CriterionScore, QualitativeResult


def _score_record(
    *,
    llm_model: str = "test-model",
    variant: str = "baseline",
    scenario_id: str = "regular",
    mae_pct: float = 10.0,
    mse_pct: float = 100.0,
    soft_aggregate: float = 0.8,
    safety_adherence: float = 1.0,
    safety_violations: list[str] | None = None,
    iterations: int = 1,
    elapsed_s: float = 1.0,
) -> ScoreRecord:
    return ScoreRecord(
        llm_model=llm_model,
        variant=variant,
        scenario_id=scenario_id,
        judge_model="test",
        mae_pct=mae_pct,
        mse_pct=mse_pct,
        per_nutrient_pct={},
        qualitative=QualitativeResult(
            scores=[
                CriterionScore(
                    criterion_id="recipe-makes-sense",
                    score=soft_aggregate,
                    reasoning="ok",
                ),
            ],
            aggregate=soft_aggregate,
            safety_adherence=safety_adherence,
            safety_violations=safety_violations or [],
        ),
        iterations=iterations,
        elapsed_s=elapsed_s,
    )


def test_summarize_averages_within_variant() -> None:
    records = [
        _score_record(scenario_id="regular", mae_pct=10.0, mse_pct=100.0, soft_aggregate=0.8, elapsed_s=2.0),
        _score_record(scenario_id="cut", mae_pct=20.0, mse_pct=200.0, soft_aggregate=0.6, elapsed_s=4.0),
    ]
    summaries = summarize(records)
    assert len(summaries) == 1
    row = summaries[0]
    assert row == VariantSummary(
        llm_model="test-model",
        variant="baseline",
        n_runs=2,
        mae_pct=15.0,
        mse_pct=150.0,
        soft_aggregate=0.7,
        safety_adherence=1.0,
        iterations=1.0,
        elapsed_s=3.0,
        n_safety_violations=0,
    )


def test_summarize_groups_by_model_and_variant() -> None:
    records = [
        _score_record(variant="baseline", llm_model="model-a"),
        _score_record(variant="full", llm_model="model-b"),
    ]
    summaries = summarize(records)
    assert len(summaries) == 2
    by_model = {row.llm_model: row for row in summaries}
    assert by_model["model-a"].variant == "baseline"
    assert by_model["model-b"].variant == "full"


def test_summarize_counts_safety_violations() -> None:
    records = [
        _score_record(safety_adherence=0.0, safety_violations=["allergen hit"]),
        _score_record(safety_adherence=0.5, safety_violations=["a", "b"]),
    ]
    summaries = summarize(records)
    assert summaries[0].n_safety_violations == 3
