"""Paired sign test across scenarios, treating each scenario as a block."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from evaluation.plotting.stats import Metric
from evaluation.records import ScoredRun


@dataclass(frozen=True)
class ScenarioDelta:
    scenario_id: str
    baseline: float
    treatment: float

    @property
    def delta(self) -> float:
        return self.treatment - self.baseline


@dataclass(frozen=True)
class SignTest:
    deltas: tuple[ScenarioDelta, ...]
    n_effective: int
    n_improved: int
    p_value: float | None


def cell_means(records: list[ScoredRun], metric: Metric) -> dict[tuple[str, str, str], float]:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for record in records:
        key = (record.variant, record.scenario_id, record.llm_model)
        buckets.setdefault(key, []).append(metric.value(record))
    return {key: statistics.fmean(values) for key, values in buckets.items()}


def _is_improved(delta: float, metric: Metric) -> bool:
    return delta > 0 if metric.higher_is_better else delta < 0


def _exact_one_sided_p(n: int, m: int) -> float:
    return sum(math.comb(n, k) for k in range(m, n + 1)) / 2**n


def paired_sign_test(
    records: list[ScoredRun],
    metric: Metric,
    *,
    baseline_variant: str,
    treatment_variant: str,
) -> SignTest:
    means = cell_means(records, metric)
    cells = sorted({(scenario, model) for variant, scenario, model in means if variant == baseline_variant})
    deltas: list[ScenarioDelta] = []
    for scenario, model in cells:
        baseline = means.get((baseline_variant, scenario, model))
        treatment = means.get((treatment_variant, scenario, model))
        if baseline is None or treatment is None:
            continue
        deltas.append(ScenarioDelta(scenario_id=scenario, baseline=baseline, treatment=treatment))

    nonzero = [pair for pair in deltas if pair.delta != 0]
    n_effective = len(nonzero)
    n_improved = sum(1 for pair in nonzero if _is_improved(pair.delta, metric))
    p_value = _exact_one_sided_p(n_effective, n_improved) if n_effective else None
    return SignTest(
        deltas=tuple(deltas),
        n_effective=n_effective,
        n_improved=n_improved,
        p_value=p_value,
    )
