"""Tests for experiment figure rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from dietary_advisor.config.llm import LlmSpec
from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.case_runner.grid import RunSpec
from evaluation.judges import all_judges
from evaluation.plotting.figures import _error_bounds, render_experiment, render_run_variability
from evaluation.plotting.stats import (
    _short_model_name,
    BY_EFFORT,
    BY_MODEL,
    complete_spec_groups,
    group_metric_stats,
    group_metric_stats_by_judge,
    grouped_metric_stats,
    GroupStats,
    ITERATIONS_METRIC,
    MAE_METRIC,
    mean_abs_deviation,
    model_variability,
    PLOT_METRICS,
    pooled_variability,
    POOLED_VARIABILITY_LABEL,
    SOFT_METRIC,
    variability_by_judge,
)
from evaluation.records import ScoredRun
from tests.evaluation.conftest import make_scored_run

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
) -> ScoredRun:
    return make_scored_run(
        llm_model=llm_model,
        variant=variant,
        scenario_id=scenario_id,
        mae_pct=mae_pct,
        soft=soft,
        safety=safety,
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
    stats = group_metric_stats(records, MAE_METRIC)

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
    stats = group_metric_stats(records, MAE_METRIC)
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
        SOFT_METRIC,
    )
    assert len(series) == 2
    assert [label for label, _ in series] == [JUDGE_A.label, JUDGE_B.label]
    assert series[0][1][0].mean == pytest.approx(0.8)
    assert series[1][1][0].mean == pytest.approx(0.6)


def test_render_experiment_writes_one_png_per_plot_metric(tmp_path: Path) -> None:
    variant = VariantConfig(totaller_enabled=False, reflection_enabled=True)
    records = [
        _score_record(variant=variant.label, scenario_id="regular", mae_pct=5.0),
        _score_record(variant=variant.label, scenario_id="vegetarian-allergic", mae_pct=7.0),
    ]
    paths = render_experiment("ablation", {JUDGE_A.key: records}, output_dir=tmp_path)

    png_names = {path.name for path in paths if path.suffix == ".png"}
    assert png_names == {metric.filename for metric in PLOT_METRICS}
    assert "safety_adherence.png" not in png_names
    assert "tradeoff_judge_1.png" not in png_names
    for path in paths:
        if path.suffix == ".png":
            assert path.parent == tmp_path / "figures" / "ablation"
            assert path.stat().st_size > 0
            assert path.with_suffix(".pdf").is_file()


def test_render_experiment_skips_iterations_without_reflection(tmp_path: Path) -> None:
    variant = VariantConfig(totaller_enabled=False, reflection_enabled=False)
    records = [
        _score_record(variant=variant.label, scenario_id="regular", mae_pct=5.0),
        _score_record(variant=variant.label, scenario_id="vegetarian-allergic", mae_pct=7.0),
    ]
    paths = render_experiment("ablation", {JUDGE_A.key: records}, output_dir=tmp_path)

    png_names = {path.name for path in paths if path.suffix == ".png"}
    assert "iterations.png" not in png_names
    assert "safety_adherence.png" not in png_names


def test_render_experiment_writes_extra_model_groupings(tmp_path: Path) -> None:
    records = [
        _score_record(llm_model="openrouter:google/gemini-3.6-flash#low", variant="totaller+reflective-loop"),
        _score_record(llm_model="openrouter:openai/gpt-5.6-luna#xhigh", variant="totaller+reflective-loop"),
    ]
    paths = render_experiment("models", {JUDGE_A.key: records}, output_dir=tmp_path)

    png_names = {path.name for path in paths if path.suffix == ".png"}
    assert "mae_pct_by_model.png" in png_names
    assert "mae_pct_by_effort.png" in png_names
    dest = tmp_path / "figures" / "models"
    assert (dest / "mae_pct_by_model.png").stat().st_size > 0
    assert (dest / "mae_pct_by_effort.png").stat().st_size > 0
    tradeoff = dest / "tradeoff_judge_1.png"
    assert tradeoff in paths
    assert tradeoff.stat().st_size > 0
    assert tradeoff.with_suffix(".pdf").is_file()
    assert "tradeoff_judge_2.png" not in png_names
    assert "tradeoff_judge_mean.png" not in png_names


def test_grouped_metric_stats_by_model_combines_efforts() -> None:
    records = [
        _score_record(llm_model="openrouter:google/gemini-3.6-flash#low", mae_pct=8.0),
        _score_record(llm_model="openrouter:google/gemini-3.6-flash#high", mae_pct=12.0),
        _score_record(llm_model="openrouter:openai/gpt-5.6-luna#low", mae_pct=4.0),
    ]
    stats = grouped_metric_stats(records, MAE_METRIC, BY_MODEL)

    assert [group.label for group in stats] == ["gemini-3.6-flash", "gpt-5.6-luna"]
    assert stats[0].n == 2
    assert stats[0].mean == pytest.approx(10.0)
    assert stats[1].n == 1
    assert stats[1].mean == pytest.approx(4.0)


def test_grouped_metric_stats_by_model_sorts_descending() -> None:
    records = [
        _score_record(llm_model="openrouter:google/gemini-3.6-flash", mae_pct=4.0),
        _score_record(llm_model="openrouter:openai/gpt-5.6-luna", mae_pct=12.0),
    ]
    stats = grouped_metric_stats(records, MAE_METRIC, BY_MODEL)
    assert [group.label for group in stats] == ["gpt-5.6-luna", "gemini-3.6-flash"]


def test_group_metric_stats_sorts_models_descending() -> None:
    records = [
        _score_record(llm_model="openrouter:google/gemini-3.6-flash", variant="totaller+reflective-loop", mae_pct=4.0),
        _score_record(llm_model="openrouter:openai/gpt-5.6-luna", variant="totaller+reflective-loop", mae_pct=12.0),
    ]
    stats = group_metric_stats(records, MAE_METRIC)
    assert [group.label for group in stats] == ["gpt-5.6-luna", "gemini-3.6-flash"]


def test_grouped_metric_stats_by_effort_maps_and_orders_buckets() -> None:
    records = [
        _score_record(llm_model="openrouter:openai/gpt-5.6-luna#xhigh", mae_pct=6.0),
        _score_record(llm_model="openrouter:google/gemini-3.5-flash-lite", mae_pct=10.0),
        _score_record(llm_model="openrouter:google/gemini-3.6-flash", mae_pct=8.0),
        _score_record(llm_model="openrouter:google/gemini-3.6-flash#low", mae_pct=12.0),
    ]
    stats = grouped_metric_stats(records, MAE_METRIC, BY_EFFORT)

    assert [group.label for group in stats] == ["low", "medium", "high"]
    assert stats[0].n == 2
    assert stats[0].mean == pytest.approx(11.0)
    assert stats[1].n == 1
    assert stats[1].mean == pytest.approx(8.0)
    assert stats[2].n == 1
    assert stats[2].mean == pytest.approx(6.0)


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
    stats = model_variability(groups, SOFT_METRIC)

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
    stats = model_variability(groups, SOFT_METRIC)

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
    stats = model_variability(groups, SOFT_METRIC)

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
    stats = pooled_variability(groups, SOFT_METRIC)

    assert stats.label == POOLED_VARIABILITY_LABEL
    assert stats.n_specs == 2
    assert stats.mean == pytest.approx(0.4 / 3 / 2)


def test_variability_by_judge_sorts_models_descending() -> None:
    model_low = "openrouter:google/gemini-3.6-flash"
    model_high = "openrouter:openai/gpt-5.6-luna"
    groups = [
        [_score_record(llm_model=model_low, soft=0.8) for _ in range(3)],
        [_score_record(llm_model=model_high, soft=0.6 + 0.1 * rep) for rep in range(3)],
    ]
    label_order, _ = variability_by_judge({JUDGE_A.key: groups}, SOFT_METRIC)

    assert label_order == [
        _short_model_name(model_high),
        _short_model_name(model_low),
        POOLED_VARIABILITY_LABEL,
    ]


def test_variability_by_judge_appends_pooled_column() -> None:
    model = "openrouter:openai/gpt-5.6-luna"
    groups = [
        [_score_record(llm_model=model, soft=0.6 + 0.1 * rep) for rep in range(3)],
    ]
    label_order, series = variability_by_judge({JUDGE_A.key: groups}, SOFT_METRIC)

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
    _, series = variability_by_judge({JUDGE_A.key: groups_j1, JUDGE_B.key: groups_j2}, SOFT_METRIC)

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


def test_group_stats_sem() -> None:
    group = GroupStats(label="test", mean=1.0, std=0.6, n=9)
    assert group.sem == pytest.approx(0.2)

    single = GroupStats(label="solo", mean=1.0, std=0.0, n=1)
    assert single.sem == pytest.approx(0.0)


def test_error_bounds_clamps_lower_whisker_at_floor() -> None:
    stats = [GroupStats(label="tiny", mean=0.1, std=2.0, n=4)]
    lower, upper = _error_bounds(stats, ITERATIONS_METRIC)

    assert upper == [pytest.approx(1.0)]
    assert lower == [pytest.approx(0.1)]


def test_error_bounds_uses_sem_when_above_floor() -> None:
    stats = [GroupStats(label="ok", mean=1.5, std=0.4, n=4)]
    lower, upper = _error_bounds(stats, ITERATIONS_METRIC)

    assert upper == [pytest.approx(0.2)]
    assert lower == [pytest.approx(0.2)]


def test_soft_metric_uses_three_decimal_places() -> None:
    assert SOFT_METRIC.decimals == 3


def test_render_experiment_writes_tradeoff_per_judge(tmp_path: Path) -> None:
    records = [
        _score_record(llm_model="openrouter:google/gemini-3.6-flash#low", variant="totaller+reflective-loop"),
        _score_record(llm_model="openrouter:openai/gpt-5.6-luna#xhigh", variant="totaller+reflective-loop"),
    ]
    paths = render_experiment(
        "models",
        {JUDGE_A.key: records, JUDGE_B.key: records},
        output_dir=tmp_path,
    )
    dest = tmp_path / "figures" / "models"
    for name in (f"tradeoff_{JUDGE_A.key}", f"tradeoff_{JUDGE_B.key}", "tradeoff_judge_mean"):
        png = dest / f"{name}.png"
        assert png in paths
        assert png.stat().st_size > 0
        assert png.with_suffix(".pdf").is_file()
