from __future__ import annotations

import pytest

from evaluation.paired import cell_means, paired_sign_test
from evaluation.plotting.stats import MAE_METRIC, SOFT_METRIC
from tests.evaluation.conftest import make_scored_run

_SCENARIOS = (
    "diabetes-hypertension",
    "dyslipidemia-obesity",
    "lactose-intolerant-athlete-wants-cheesecake",
    "regular",
    "vegetarian-allergic",
)


def _ablation_grid(*, baseline_mae: list[float], totaller_mae: list[float]) -> list:
    records = []
    for scenario, base, totaller in zip(_SCENARIOS, baseline_mae, totaller_mae, strict=True):
        records.append(make_scored_run(variant="baseline", scenario_id=scenario, mae_pct=base))
        records.append(make_scored_run(variant="totaller", scenario_id=scenario, mae_pct=totaller))
    return records


def test_sign_test_five_of_five_mae_improvements() -> None:
    records = _ablation_grid(
        baseline_mae=[4.3, 7.4, 34.1, 8.1, 13.4],
        totaller_mae=[3.3, 3.2, 3.8, 2.1, 2.7],
    )
    result = paired_sign_test(
        records,
        MAE_METRIC,
        baseline_variant="baseline",
        treatment_variant="totaller",
    )

    assert result.n_effective == 5
    assert result.n_improved == 5
    assert result.p_value == pytest.approx(1 / 32)


def test_sign_test_mixed_signs_is_inconclusive() -> None:
    records = _ablation_grid(
        baseline_mae=[3.3, 3.2, 3.8, 2.1, 2.7],
        totaller_mae=[1.5, 5.1, 3.7, 3.5, 2.5],
    )
    result = paired_sign_test(
        records,
        MAE_METRIC,
        baseline_variant="baseline",
        treatment_variant="totaller",
    )

    assert result.n_effective == 5
    assert result.n_improved == 3
    assert result.p_value == pytest.approx(0.5)


def test_sign_test_drops_ties_from_n() -> None:
    records = _ablation_grid(
        baseline_mae=[4.0, 5.0, 6.0, 7.0, 8.0],
        totaller_mae=[3.0, 4.0, 5.0, 6.0, 8.0],
    )
    result = paired_sign_test(
        records,
        MAE_METRIC,
        baseline_variant="baseline",
        treatment_variant="totaller",
    )

    assert result.n_effective == 4
    assert result.n_improved == 4
    assert result.p_value == pytest.approx(1 / 16)


def test_sign_test_pairs_on_cell_means_not_runs() -> None:
    records = []
    for mae in (10.0, 20.0, 30.0):
        records.append(make_scored_run(variant="baseline", scenario_id="regular", mae_pct=mae))
    for mae in (1.0, 2.0, 3.0):
        records.append(make_scored_run(variant="totaller", scenario_id="regular", mae_pct=mae))
    means = cell_means(records, MAE_METRIC)
    result = paired_sign_test(
        records,
        MAE_METRIC,
        baseline_variant="baseline",
        treatment_variant="totaller",
    )

    assert means[("baseline", "regular", "openrouter:openai/gpt-5.6-luna")] == pytest.approx(20.0)
    assert means[("totaller", "regular", "openrouter:openai/gpt-5.6-luna")] == pytest.approx(2.0)
    assert len(result.deltas) == 1
    assert result.deltas[0].delta == pytest.approx(-18.0)
    assert result.n_improved == 1


def test_soft_metric_counts_an_increase_as_improvement() -> None:
    records = []
    for scenario in _SCENARIOS:
        records.append(make_scored_run(variant="totaller", scenario_id=scenario, soft=0.70))
        records.append(make_scored_run(variant="totaller+reflective-loop", scenario_id=scenario, soft=0.80))
    result = paired_sign_test(
        records,
        SOFT_METRIC,
        baseline_variant="totaller",
        treatment_variant="totaller+reflective-loop",
    )

    assert result.n_improved == 5
    assert result.p_value == pytest.approx(1 / 32)


def test_mae_metric_counts_a_decrease_as_improvement() -> None:
    records = _ablation_grid(
        baseline_mae=[10.0, 10.0, 10.0, 10.0, 10.0],
        totaller_mae=[11.0, 11.0, 11.0, 11.0, 11.0],
    )
    result = paired_sign_test(
        records,
        MAE_METRIC,
        baseline_variant="baseline",
        treatment_variant="totaller",
    )

    assert result.n_improved == 0
    assert result.p_value == pytest.approx(1.0)


def test_sign_test_all_ties_leaves_p_undefined() -> None:
    records = _ablation_grid(
        baseline_mae=[4.0, 5.0, 6.0, 7.0, 8.0],
        totaller_mae=[4.0, 5.0, 6.0, 7.0, 8.0],
    )
    result = paired_sign_test(
        records,
        MAE_METRIC,
        baseline_variant="baseline",
        treatment_variant="totaller",
    )

    assert result.n_effective == 0
    assert result.n_improved == 0
    assert result.p_value is None


def test_sign_test_is_one_sided() -> None:
    records = _ablation_grid(
        baseline_mae=[4.3, 7.4, 34.1, 8.1, 13.4],
        totaller_mae=[3.3, 3.2, 3.8, 2.1, 2.7],
    )
    result = paired_sign_test(
        records,
        MAE_METRIC,
        baseline_variant="baseline",
        treatment_variant="totaller",
    )

    assert result.p_value == pytest.approx(1 / 32)
    assert result.p_value != pytest.approx(1 / 16)


def test_cell_means_keeps_models_apart() -> None:
    records = [
        make_scored_run(llm_model="model-a", variant="baseline", scenario_id="regular", mae_pct=10.0),
        make_scored_run(llm_model="model-b", variant="baseline", scenario_id="regular", mae_pct=30.0),
    ]
    means = cell_means(records, MAE_METRIC)

    assert means[("baseline", "regular", "model-a")] == pytest.approx(10.0)
    assert means[("baseline", "regular", "model-b")] == pytest.approx(30.0)
