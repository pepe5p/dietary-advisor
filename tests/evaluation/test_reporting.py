"""Tests for experiment metrics JSON export."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.plotting.stats import group_metric_stats, METRICS
from evaluation.reporting import ExperimentMetrics, write_experiment_metrics
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


def test_write_experiment_metrics_writes_expected_path(tmp_path: Path) -> None:
    records = [
        _score_record(variant="baseline", scenario_id="regular", mae_pct=8.0),
        _score_record(variant="baseline", scenario_id="vegetarian-allergic", mae_pct=12.0),
        _score_record(variant="totaller", scenario_id="regular", mae_pct=6.0),
    ]
    path = write_experiment_metrics("ablation", records, output_dir=tmp_path)

    assert path == tmp_path / "metrics" / "ablation.json"
    assert path.is_file()


def test_write_experiment_metrics_round_trips_and_matches_group_stats(tmp_path: Path) -> None:
    records = [
        _score_record(variant="baseline", scenario_id="regular", mae_pct=8.0),
        _score_record(variant="baseline", scenario_id="vegetarian-allergic", mae_pct=12.0),
        _score_record(variant="totaller", scenario_id="regular", mae_pct=6.0),
    ]
    path = write_experiment_metrics("ablation", records, output_dir=tmp_path)
    report = ExperimentMetrics.model_validate_json(path.read_text(encoding="utf-8"))

    assert report.experiment == "ablation"
    assert report.n_runs == 3
    assert len(report.metrics) == len(METRICS)

    mae_metric = next(metric for metric in METRICS if metric.key == "mae_pct")
    expected_groups = group_metric_stats(records, mae_metric)
    assert report.metrics["mae_pct"].groups == expected_groups
    assert report.metrics["mae_pct"].title == mae_metric.title
    assert report.metrics["mae_pct"].ylabel == mae_metric.ylabel


def test_write_experiment_metrics_includes_summaries_per_model_variant(tmp_path: Path) -> None:
    records = [
        _score_record(variant="baseline", scenario_id="regular", mae_pct=8.0),
        _score_record(variant="baseline", scenario_id="vegetarian-allergic", mae_pct=12.0),
        _score_record(variant="totaller", scenario_id="regular", mae_pct=6.0),
    ]
    path = write_experiment_metrics("ablation", records, output_dir=tmp_path)
    report = ExperimentMetrics.model_validate_json(path.read_text(encoding="utf-8"))

    assert {(row.variant, row.n_runs) for row in report.summaries} == {
        ("baseline", 2),
        ("totaller", 1),
    }
    baseline = next(row for row in report.summaries if row.variant == "baseline")
    assert baseline.mae_pct == pytest.approx(10.0)
    assert baseline.n_safety_violations == 0
