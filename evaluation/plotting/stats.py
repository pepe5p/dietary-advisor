from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from dietary_advisor.config.llm import LlmSpec
from evaluation.case_runner.grid import MODEL_COMPARISON_MODELS, RunSpec, VARIANTS
from evaluation.judges import all_judges
from evaluation.scoring.store import ScoreRecord


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    ylabel: str
    value: Callable[[ScoreRecord], float]
    judge_dependent: bool = False

    @property
    def filename(self) -> str:
        return f"{self.key}.png"


class GroupStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    mean: float
    std: float
    n: int


class ModelVariability(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    mean: float
    n_specs: int


SPREAD_COLUMN_LABEL = "Spread (mean |run - spec mean|)"
POOLED_VARIABILITY_LABEL = "all"

METRICS: tuple[Metric, ...] = (
    Metric("mae_pct", "Macro MAE %", "MAE %", lambda record: record.mae_pct),
    Metric(
        "soft_aggregate",
        "Soft preference aggregate",
        "Score",
        lambda record: record.qualitative.aggregate,
        judge_dependent=True,
    ),
    Metric(
        "safety_adherence",
        "Safety adherence",
        "Score",
        lambda record: record.qualitative.safety_adherence,
        judge_dependent=True,
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


def primary_records(records_by_judge: dict[str, list[ScoreRecord]]) -> list[ScoreRecord]:
    """Records of the first judge that has any, for metrics that do not depend on the judge."""
    for judge in all_judges():
        records = records_by_judge.get(judge.key)
        if records:
            return records
    return []


def group_metric_stats_by_judge(
    records_by_judge: dict[str, list[ScoreRecord]],
    metric: Metric,
) -> list[tuple[str, list[GroupStats]]]:
    """Per-judge group stats, every judge aligned to the first judge's group order."""
    primary = primary_records(records_by_judge)
    if not primary:
        return []

    label_order = [group.label for group in group_metric_stats(primary, metric)]
    series: list[tuple[str, list[GroupStats]]] = []

    for judge in all_judges():
        records = records_by_judge.get(judge.key)
        if not records:
            continue
        by_label = {group.label: group for group in group_metric_stats(records, metric)}
        aligned = [by_label[label] for label in label_order if label in by_label]
        if aligned:
            series.append((judge.label, aligned))

    return series


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


def mean_abs_deviation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.fmean([abs(value - mean) for value in values])


def model_variability(groups: list[list[ScoreRecord]], metric: Metric) -> list[ModelVariability]:
    by_model: dict[str, list[float]] = {}
    for group in groups:
        spread = mean_abs_deviation([metric.value(record) for record in group])
        model = group[0].llm_model
        by_model.setdefault(model, []).append(spread)

    stats: list[ModelVariability] = []
    for model in _model_order(set(by_model)):
        spreads = by_model[model]
        stats.append(
            ModelVariability(
                label=_short_model_name(model),
                mean=statistics.fmean(spreads),
                n_specs=len(spreads),
            ),
        )
    return stats


def pooled_variability(groups: list[list[ScoreRecord]], metric: Metric) -> ModelVariability:
    spreads = [mean_abs_deviation([metric.value(record) for record in group]) for group in groups]
    return ModelVariability(
        label=POOLED_VARIABILITY_LABEL,
        mean=statistics.fmean(spreads),
        n_specs=len(spreads),
    )


def variability_by_judge(
    groups_by_judge: dict[str, list[list[ScoreRecord]]],
    metric: Metric,
) -> tuple[list[str], list[tuple[str, list[ModelVariability | None]]]]:
    all_models: set[str] = set()
    per_judge: dict[str, dict[str, ModelVariability]] = {}
    groups_by_judge_key: dict[str, list[list[ScoreRecord]]] = {}

    for judge in all_judges():
        groups = groups_by_judge.get(judge.key)
        if not groups:
            continue
        groups_by_judge_key[judge.key] = groups
        stats = model_variability(groups, metric)
        if not stats:
            continue
        per_judge[judge.key] = {entry.label: entry for entry in stats}
        for group in groups:
            all_models.add(group[0].llm_model)

    if not per_judge:
        return [], []

    label_order = [_short_model_name(model) for model in _model_order(all_models)]
    label_order.append(POOLED_VARIABILITY_LABEL)
    series: list[tuple[str, list[ModelVariability | None]]] = []
    for judge in all_judges():
        groups = groups_by_judge_key.get(judge.key)
        by_label = per_judge.get(judge.key)
        if not groups or not by_label:
            continue
        aligned = [by_label.get(label) for label in label_order[:-1]]
        aligned.append(pooled_variability(groups, metric))
        series.append((judge.label, aligned))

    return label_order, series
