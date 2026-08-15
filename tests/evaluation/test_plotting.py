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
    mean_abs_deviation,
    METRICS,
    model_variability,
    pooled_variability,
    POOLED_VARIABILITY_LABEL,
    variability_by_judge,
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


def test_mean_abs_deviation_averages_distance_from_the_mean() -> None:
    assert mean_abs_deviation([0.6, 0.8, 1.0]) == pytest.approx(0.4 / 3)


def test_mean_abs_deviation_is_zero_for_identical_values() -> None:
    assert mean_abs_deviation([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_model_variability_averages_spreads_within_model() -> None:
    model_a = "openrouter:openai/gpt-5.6-luna"
    model_b = "openrouter:google/gemini-3.6-flash"
    groups = [
        [
            _score_record(llm_model=model_a, soft=0.6),
            _score_record(llm_model=model_a, soft=0.8),
            _score_record(llm_model=model_a, soft=1.0),
        ],
        [
            _score_record(llm_model=model_b, soft=0.8),
            _score_record(llm_model=model_b, soft=0.8),
            _score_record(llm_model=model_b, soft=0.8),
        ],
    ]
    stats = model_variability(groups, METRICS[1])

    assert len(stats) == 2
    assert stats[0].label == _short_model_name(model_b)
    assert stats[0].mean == pytest.approx(0.0)
    assert stats[0].n_specs == 1
    assert stats[1].label == _short_model_name(model_a)
    assert stats[1].mean == pytest.approx(0.4 / 3)
    assert stats[1].n_specs == 1


def test_model_variability_orders_models_by_grid_definition() -> None:
    model_a = "openrouter:google/gemini-3.5-flash-lite"
    model_b = "openrouter:google/gemini-3.6-flash"
    groups = [
        [_score_record(llm_model=model_b, soft=0.7 + 0.1 * rep) for rep in range(3)],
        [_score_record(llm_model=model_a, soft=0.7 + 0.1 * rep) for rep in range(3)],
    ]
    stats = model_variability(groups, METRICS[1])

    assert [entry.label for entry in stats] == [
        _short_model_name(model_a),
        _short_model_name(model_b),
    ]


def test_model_variability_averages_zero_spread_specs() -> None:
    model = "openrouter:openai/gpt-5.6-luna"
    groups = [
        [
            _score_record(llm_model=model, soft=0.0),
            _score_record(llm_model=model, soft=0.0),
            _score_record(llm_model=model, soft=0.0),
        ],
        [
            _score_record(llm_model=model, soft=0.6),
            _score_record(llm_model=model, soft=0.8),
            _score_record(llm_model=model, soft=1.0),
        ],
    ]
    stats = model_variability(groups, METRICS[1])

    assert len(stats) == 1
    assert stats[0].n_specs == 2
    assert stats[0].mean == pytest.approx(0.4 / 3 / 2)


def test_pooled_variability_averages_all_specs() -> None:
    model_a = "openrouter:openai/gpt-5.6-luna"
    model_b = "openrouter:google/gemini-3.6-flash"
    groups = [
        [
            _score_record(llm_model=model_a, soft=0.6),
            _score_record(llm_model=model_a, soft=0.8),
            _score_record(llm_model=model_a, soft=1.0),
        ],
        [
            _score_record(llm_model=model_b, soft=0.8),
            _score_record(llm_model=model_b, soft=0.8),
            _score_record(llm_model=model_b, soft=0.8),
        ],
    ]
    stats = pooled_variability(groups, METRICS[1])

    assert stats.label == POOLED_VARIABILITY_LABEL
    assert stats.n_specs == 2
    assert stats.mean == pytest.approx(0.4 / 3 / 2)


def test_variability_by_judge_appends_pooled_column() -> None:
    model = "openrouter:openai/gpt-5.6-luna"
    groups = [
        [_score_record(llm_model=model, soft=0.6 + 0.1 * rep) for rep in range(3)],
    ]
    label_order, series = variability_by_judge({JUDGE_A.key: groups}, METRICS[1])

    assert label_order[-1] == POOLED_VARIABILITY_LABEL
    pooled = series[0][1][-1]
    assert pooled is not None
    assert pooled.mean == pytest.approx(mean_abs_deviation([0.6, 0.7, 0.8]))


def test_variability_by_judge_counts_specs_per_judge() -> None:
    model = "openrouter:openai/gpt-5.6-luna"
    groups_j1 = [
        [_score_record(llm_model=model, scenario_id=scenario, soft=0.6 + 0.1 * rep) for rep in range(3)]
        for scenario in ("regular", "vegetarian-allergic")
    ]
    groups_j2 = [
        [_score_record(llm_model=model, scenario_id="regular", soft=0.6 + 0.1 * rep) for rep in range(3)],
    ]
    _, series = variability_by_judge({JUDGE_A.key: groups_j1, JUDGE_B.key: groups_j2}, METRICS[1])

    n_specs_by_judge = {label: [entry.n_specs for entry in stats if entry] for label, stats in series}
    assert n_specs_by_judge[JUDGE_A.label] == [2, 2]
    assert n_specs_by_judge[JUDGE_B.label] == [1, 1]


def test_render_run_variability_writes_png_and_pdf(tmp_path: Path) -> None:
    model_a = "openrouter:openai/gpt-5.6-luna"
    model_b = "openrouter:google/gemini-3.6-flash"
    groups_j1 = [
        [
            _score_record(
                llm_model=model_a,
                scenario_id="regular",
                mae_pct=5.0 + rep,
                soft=0.7 + 0.1 * rep,
            )
            for rep in range(3)
        ],
        [
            _score_record(
                llm_model=model_b,
                scenario_id="diabetes-hypertension",
                mae_pct=3.0 + rep,
                soft=0.8,
            )
            for rep in range(3)
        ],
    ]
    groups_j2 = [
        [
            _score_record(
                llm_model=model_a,
                scenario_id="regular",
                mae_pct=5.0 + rep,
                soft=0.5 + 0.1 * rep,
            )
            for rep in range(3)
        ],
    ]
    paths = render_run_variability(
        {JUDGE_A.key: groups_j1, JUDGE_B.key: groups_j2},
        output_dir=tmp_path,
    )

    png_path = tmp_path / "figures" / "stability" / "run_variability.png"
    assert paths[0] == png_path
    assert png_path.stat().st_size > 0
    assert png_path.with_suffix(".pdf").is_file()
