from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from dietary_advisor.config.llm import LlmSpec
from evaluation.case_runner.grid import MINIMAL_SCENARIOS, MODEL_COMPARISON_MODELS, RunSpec, VARIANTS
from evaluation.scoring.store import ScoreRecord


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    ylabel: str
    value: Callable[[ScoreRecord], float]

    @property
    def filename(self) -> str:
        return f"{self.key}.png"


class GroupStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    mean: float
    std: float
    n: int


class SpreadStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    n_specs: int
    grand_mean: float
    deviations: list[float]
    per_spec_std: list[float]
    pooled_std: float


METRICS: tuple[Metric, ...] = (
    Metric("mae_pct", "Macro MAE %", "MAE %", lambda record: record.mae_pct),
    Metric(
        "soft_aggregate",
        "Soft preference aggregate",
        "Score",
        lambda record: record.qualitative.aggregate,
    ),
    Metric(
        "safety_adherence",
        "Safety adherence",
        "Score",
        lambda record: record.qualitative.safety_adherence,
    ),
    Metric("iterations", "Iterations", "Count", lambda record: float(record.iterations)),
    Metric("elapsed_s", "Elapsed time", "Seconds", lambda record: record.elapsed_s),
)


def _short_model_name(model: str) -> str:
    spec = LlmSpec.parse(model)
    short = spec.model.rsplit("/", 1)[-1]
    if spec.reasoning is None:
        return short
    return f"{short} ({spec.reasoning})"


def _bare_short_model_name(model: str) -> str:
    return LlmSpec.parse(model).model.rsplit("/", 1)[-1]


def _variant_order(variants: set[str]) -> list[str]:
    preferred = [variant.label for variant in VARIANTS]
    ordered = [label for label in preferred if label in variants]
    return ordered + sorted(variants - set(ordered))


def _model_order(models: set[str]) -> list[str]:
    ordered = [str(spec) for spec in MODEL_COMPARISON_MODELS if str(spec) in models]
    return ordered + sorted(models - set(ordered))


def _group_records(records: list[ScoreRecord]) -> dict[tuple[str, str], list[ScoreRecord]]:
    groups: dict[tuple[str, str], list[ScoreRecord]] = {}
    for record in records:
        key = (record.llm_model, record.variant)
        groups.setdefault(key, []).append(record)
    return groups


def group_metric_stats(records: list[ScoreRecord], metric: Metric) -> list[GroupStats]:
    groups = _group_records(records)
    models = {model for model, _ in groups}
    variants = {variant for _, variant in groups}

    if len(models) == 1:
        model = next(iter(models))
        ordered_variants = _variant_order(variants)
        keys = [(model, variant) for variant in ordered_variants]
        labels = ordered_variants
    elif len(variants) == 1:
        variant = next(iter(variants))
        ordered_models = _model_order(models)
        keys = [(model, variant) for model in ordered_models]
        labels = [_short_model_name(model) for model in ordered_models]
    else:
        keys = sorted(groups)
        labels = [f"{_short_model_name(model)} / {variant}" for model, variant in keys]

    stats: list[GroupStats] = []
    for label, key in zip(labels, keys, strict=True):
        values = [metric.value(record) for record in groups[key]]
        stats.append(
            GroupStats(
                label=label,
                mean=statistics.fmean(values),
                std=statistics.pstdev(values) if len(values) > 1 else 0.0,
                n=len(values),
            ),
        )
    return stats


def _short_scenario_name(scenario_id: str) -> str:
    parts = scenario_id.split("-")
    return "-".join(parts[:2])


def _scenario_order(scenarios: set[str]) -> list[str]:
    ordered = [scenario for scenario in MINIMAL_SCENARIOS if scenario in scenarios]
    return ordered + sorted(scenarios - set(ordered))


def complete_spec_groups(
    pairs: list[tuple[RunSpec, ScoreRecord]],
    *,
    reps: int = 3,
) -> list[list[ScoreRecord]]:
    groups: dict[tuple[str, str, str], dict[int, ScoreRecord]] = {}
    for spec, record in pairs:
        key = (spec.llm.name, spec.variant.label, spec.scenario_id)
        groups.setdefault(key, {})[spec.rep] = record

    result: list[list[ScoreRecord]] = []
    for by_rep in groups.values():
        if set(by_rep) != set(range(reps)):
            continue
        result.append([by_rep[rep] for rep in range(reps)])
    return result


def rep_spread_stats(
    groups: list[list[ScoreRecord]],
    metric: Metric,
    *,
    label: str,
) -> SpreadStats:
    per_spec_std: list[float] = []
    deviations: list[float] = []
    spec_means: list[float] = []

    for group in groups:
        values = [metric.value(record) for record in group]
        mean = statistics.fmean(values)
        spec_means.append(mean)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        per_spec_std.append(std)
        deviations.extend(value - mean for value in values)

    pooled_std = statistics.fmean([std**2 for std in per_spec_std]) ** 0.5 if per_spec_std else 0.0
    return SpreadStats(
        label=label,
        n_specs=len(groups),
        grand_mean=statistics.fmean(spec_means) if spec_means else 0.0,
        deviations=deviations,
        per_spec_std=per_spec_std,
        pooled_std=pooled_std,
    )


def scenario_spread_stats(
    groups: list[list[ScoreRecord]],
    metric: Metric,
) -> list[SpreadStats]:
    by_scenario: dict[str, list[list[ScoreRecord]]] = {}
    for group in groups:
        scenario_id = group[0].scenario_id
        by_scenario.setdefault(scenario_id, []).append(group)

    stats = [
        rep_spread_stats(by_scenario[scenario_id], metric, label=_short_scenario_name(scenario_id))
        for scenario_id in _scenario_order(set(by_scenario))
    ]
    stats.append(rep_spread_stats(groups, metric, label="all"))
    return stats


def sem_for_reps(pooled_std: float, n: int) -> float:
    return pooled_std / (n**0.5)
