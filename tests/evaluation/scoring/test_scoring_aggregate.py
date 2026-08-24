"""Tests for variant-level score aggregation."""

from __future__ import annotations

from evaluation.scoring.aggregate import summarize, VariantSummary
from tests.evaluation.conftest import make_scored_run


def test_summarize_averages_within_variant() -> None:
    records = [
        make_scored_run(scenario_id="regular", mae_pct=10.0, mse_pct=100.0, soft=0.8, elapsed_s=2.0),
        make_scored_run(scenario_id="cut", mae_pct=20.0, mse_pct=200.0, soft=0.6, elapsed_s=4.0),
    ]
    summaries = summarize(records)
    assert len(summaries) == 1
    row = summaries[0]
    assert row == VariantSummary(
        llm_model="openrouter:openai/gpt-5.6-luna",
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
        make_scored_run(variant="baseline", llm_model="model-a"),
        make_scored_run(variant="full", llm_model="model-b"),
    ]
    summaries = summarize(records)
    assert len(summaries) == 2
    by_model = {row.llm_model: row for row in summaries}
    assert by_model["model-a"].variant == "baseline"
    assert by_model["model-b"].variant == "full"


def test_summarize_counts_safety_violations() -> None:
    records = [
        make_scored_run(safety=0.0, safety_violations=["allergen hit"]),
        make_scored_run(safety=0.5, safety_violations=["a", "b"]),
    ]
    summaries = summarize(records)
    assert summaries[0].n_safety_violations == 3
