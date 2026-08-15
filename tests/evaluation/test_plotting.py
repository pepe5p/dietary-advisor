"""Tests for experiment figure rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from dietary_advisor.config.llm import LlmSpec
from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.case_runner.grid import RunSpec
from evaluation.judges import all_judges
from evaluation.plotting.figures import render_experiment, render_run_variability
from evaluation.plotting.stats import (
    _short_model_name,
    complete_spec_groups,
    group_metric_stats,
    group_metric_stats_by_judge,
    METRICS,
    rep_spread_stats,
    scenario_spread_stats,
    sem_for_reps,
)
from evaluation.scoring.store import ScoreRecord
from evaluation.validation.qualitative import QualitativeResult

JUDGE_A, JUDGE_B = all_judges()[0], all_judges()[1]


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


def test_short_model_name_renders_effort_suffix() -> None:
    assert _short_model_name("openrouter:openai/gpt-5.6-luna") == "gpt-5.6-luna"
    assert _short_model_name("openrouter:openai/gpt-5.6-luna#xhigh") == "gpt-5.6-luna (xhigh)"


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


def test_group_metric_stats_by_judge_aligns_labels() -> None:
    records_j1 = [
        _score_record(variant="baseline", soft=0.8),
        _score_record(variant="totaller", soft=0.9),
    ]
    records_j2 = [
        _score_record(variant="baseline", soft=0.6),
        _score_record(variant="totaller", soft=0.7),
    ]
    series = group_metric_stats_by_judge(
        {JUDGE_A.key: records_j1, JUDGE_B.key: records_j2},
        METRICS[1],
    )
    assert len(series) == 2
    assert [label for label, _ in series] == [JUDGE_A.label, JUDGE_B.label]
    assert series[0][1][0].mean == pytest.approx(0.8)
    assert series[1][1][0].mean == pytest.approx(0.6)


def test_render_experiment_writes_one_png_per_metric(tmp_path: Path) -> None:
    variant = VariantConfig(totaller_enabled=False, reflection_enabled=False)
    records = [
        _score_record(variant=variant.label, scenario_id="regular", mae_pct=5.0),
        _score_record(variant=variant.label, scenario_id="vegetarian-allergic", mae_pct=7.0),
    ]
    records_by_judge = {JUDGE_A.key: records}
    paths = render_experiment("ablation", records_by_judge, output_dir=tmp_path)

    png_paths = [path for path in paths if path.suffix == ".png"]
    assert len(png_paths) == len(METRICS)
    for path in png_paths:
        assert path.parent == tmp_path / "figures" / "ablation"
        assert path.stat().st_size > 0
        assert path.with_suffix(".pdf").is_file()


def _run_spec(*, scenario_id: str = "regular", rep: int = 0) -> RunSpec:
    return RunSpec(
        llm=LlmSpec(model="openrouter:openai/gpt-5.6-luna"),
        variant=VariantConfig(totaller_enabled=False, reflection_enabled=False),
        scenario_id=scenario_id,
        rep=rep,
    )


def test_complete_spec_groups_drops_incomplete_specs() -> None:
    complete = [
        (_run_spec(scenario_id="regular", rep=rep), _score_record(scenario_id="regular", mae_pct=10.0 + rep))
        for rep in range(3)
    ]
    incomplete = [
        (_run_spec(scenario_id="vegetarian-allergic", rep=0), _score_record(scenario_id="vegetarian-allergic")),
        (_run_spec(scenario_id="vegetarian-allergic", rep=2), _score_record(scenario_id="vegetarian-allergic")),
    ]
    groups = complete_spec_groups(complete + incomplete)

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_rep_spread_stats_centers_runs_on_spec_mean() -> None:
    group = [
        _score_record(mae_pct=8.0),
        _score_record(mae_pct=10.0),
        _score_record(mae_pct=12.0),
    ]
    stats = rep_spread_stats([group], METRICS[0], label="regular")

    assert stats.grand_mean == pytest.approx(10.0)
    assert stats.per_spec_std == [pytest.approx(2.0)]
    assert stats.pooled_std == pytest.approx(2.0)
    assert stats.deviations == [pytest.approx(-2.0), pytest.approx(0.0), pytest.approx(2.0)]


def test_scenario_spread_stats_orders_scenarios_and_separates_noise() -> None:
    regular_group = [
        _score_record(scenario_id="regular", mae_pct=8.0),
        _score_record(scenario_id="regular", mae_pct=10.0),
        _score_record(scenario_id="regular", mae_pct=12.0),
    ]
    diabetes_group = [
        _score_record(scenario_id="diabetes-hypertension", mae_pct=4.0),
        _score_record(scenario_id="diabetes-hypertension", mae_pct=4.0),
        _score_record(scenario_id="diabetes-hypertension", mae_pct=4.0),
    ]
    stats = scenario_spread_stats([regular_group, diabetes_group], METRICS[0])

    assert [entry.label for entry in stats] == ["regular", "diabetes-hypertension", "all"]
    assert stats[0].pooled_std == pytest.approx(2.0)
    assert stats[1].pooled_std == pytest.approx(0.0)
    assert stats[2].pooled_std == pytest.approx(2.0**0.5)


def test_sem_for_reps_shrinks_with_sqrt_n() -> None:
    assert sem_for_reps(2.0, 1) == pytest.approx(2.0)
    assert sem_for_reps(2.0, 2) == pytest.approx(2.0 / 2**0.5)


def test_render_run_variability_writes_png_and_pdf(tmp_path: Path) -> None:
    groups_j1 = [
        [_score_record(scenario_id="regular", mae_pct=5.0 + rep, soft=0.7 + 0.1 * rep) for rep in range(3)],
        [_score_record(scenario_id="diabetes-hypertension", mae_pct=3.0 + rep, soft=0.8) for rep in range(3)],
    ]
    groups_j2 = [
        [_score_record(scenario_id="regular", mae_pct=5.0 + rep, soft=0.5 + 0.1 * rep) for rep in range(3)],
    ]
    paths = render_run_variability(
        {JUDGE_A.key: groups_j1, JUDGE_B.key: groups_j2},
        output_dir=tmp_path,
    )

    png_path = tmp_path / "figures" / "stability" / "run_variability.png"
    assert paths[0] == png_path
    assert png_path.stat().st_size > 0
    assert png_path.with_suffix(".pdf").is_file()
