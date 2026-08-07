"""Tests for experiment figure rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.plotting.figures import group_metric_stats, METRICS, render_experiment
from evaluation.scoring.store import ScoreRecord
from evaluation.validation.qualitative import QualitativeResult


def _score_record(
    *,
    llm_model: str = "openrouter:openai/gpt-5.6-luna",
    variant: str = "baseline",
    scenario_id: str = "regular",
    mae_pct: float = 10.0,
    soft: float = 0.8,
    safety: float = 1.0,
    iterations: int = 1,
    elapsed_s: float = 2.0,
) -> ScoreRecord:
    return ScoreRecord(
        llm_model=llm_model,
        variant=variant,
        scenario_id=scenario_id,
        judge_model="test-judge",
        mae_pct=mae_pct,
        mse_pct=mae_pct**2,
        per_nutrient_pct={"energy_kcal": mae_pct},
        qualitative=QualitativeResult(
            aggregate=soft,
            safety_adherence=safety,
        ),
        iterations=iterations,
        elapsed_s=elapsed_s,
    )


def test_group_metric_stats_computes_mean_and_std_for_reps() -> None:
    records = [
        _score_record(variant="baseline", scenario_id="regular", mae_pct=8.0),
        _score_record(variant="baseline", scenario_id="vegetarian-allergic", mae_pct=12.0),
        _score_record(variant="totaller", scenario_id="regular", mae_pct=6.0),
    ]
    stats = group_metric_stats(records, METRICS[0])

    assert [group.label for group in stats] == ["baseline", "totaller"]
    baseline = stats[0]
    assert baseline.n == 2
    assert baseline.mean == pytest.approx(10.0)
    assert baseline.std == pytest.approx(2.0)


def test_group_metric_stats_orders_variants_by_grid_definition() -> None:
    records = [
        _score_record(variant="totaller+reflective-loop", mae_pct=1.0),
        _score_record(variant="baseline", mae_pct=2.0),
        _score_record(variant="totaller", mae_pct=3.0),
        _score_record(variant="reflective-loop", mae_pct=4.0),
    ]
    stats = group_metric_stats(records, METRICS[0])
    assert [group.label for group in stats] == [
        "baseline",
        "totaller",
        "reflective-loop",
        "totaller+reflective-loop",
    ]


def test_render_experiment_writes_one_png_per_metric(tmp_path: Path) -> None:
    variant = VariantConfig(totaller_enabled=False, reflection_enabled=False)
    records = [
        _score_record(variant=variant.label, scenario_id="regular", mae_pct=5.0),
        _score_record(variant=variant.label, scenario_id="vegetarian-allergic", mae_pct=7.0),
    ]
    paths = render_experiment("ablation", records, output_dir=tmp_path)

    assert len(paths) == len(METRICS)
    for path in paths:
        assert path.parent == tmp_path / "figures" / "ablation"
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
