from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from dietary_advisor.config.llm import LlmSpec
from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.case_runner.grid import DEFAULT_EFFORT, MODEL_COMPARISON_MODELS, RunSpec, VARIANTS
from evaluation.judges import all_judges
from evaluation.records import ScoredRun

_REFLECTION_LABELS = {
    VariantConfig(totaller_enabled=totaller, rag_enabled=rag, reflection_enabled=True).label
    for totaller in (False, True)
    for rag in (False, True)
}
_EFFORT_BUCKETS = ("low", "medium", "high")
_EFFORT_TO_BUCKET = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    ylabel: str
    value: Callable[[ScoredRun], float]
    judge_dependent: bool = False
    # Soft scores cluster near 0.8; a zero-based axis hides between-group differences.
    zoom_ylim: bool = False
    applies_to: Callable[[ScoredRun], bool] | None = None
    # Every metric here is a non-negative quantity, so error bars are clamped
    # at this value rather than implying impossible readings.
    floor: float | None = 0.0
    decimals: int = 2

    @property
    def filename(self) -> str:
        return f"{self.key}.png"


@dataclass(frozen=True)
class Grouping:
    key: str
    title: str
    label: Callable[[ScoredRun], str]
    order: Callable[[set[str]], list[str]]
    # Categories with no natural order (models) are ranked by the plotted mean.
    sort_desc: bool = False


class GroupStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    mean: float
    std: float
    n: int

    @property
    def sem(self) -> float:
        return self.std / math.sqrt(self.n) if self.n > 1 else 0.0


class ModelVariability(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    mean: float
    n_specs: int


SPREAD_COLUMN_LABEL = "Spread (mean |run - spec mean|)"
POOLED_VARIABILITY_LABEL = "all"


def _has_reflection(record: ScoredRun) -> bool:
    return record.variant in _REFLECTION_LABELS


MAE_METRIC = Metric("mae_pct", "Energy and macronutrient MAE %", "MAE %", lambda record: record.mae_pct)
SOFT_METRIC = Metric(
    "soft_aggregate",
    "Soft preference aggregate",
    "Score",
    lambda record: record.qualitative.aggregate,
    judge_dependent=True,
    zoom_ylim=True,
    decimals=3,
)
SAFETY_METRIC = Metric(
    "safety_adherence",
    "Safety adherence",
    "Score",
    lambda record: record.qualitative.safety_adherence,
    judge_dependent=True,
)
ITERATIONS_METRIC = Metric(
    "iterations",
    "Iterations",
    "Count",
    lambda record: float(record.iterations),
    applies_to=_has_reflection,
)
ELAPSED_METRIC = Metric("elapsed_s", "Elapsed time", "Seconds", lambda record: record.elapsed_s)

METRICS: tuple[Metric, ...] = (
    MAE_METRIC,
    SOFT_METRIC,
    SAFETY_METRIC,
    ITERATIONS_METRIC,
    ELAPSED_METRIC,
)
PLOT_METRICS: tuple[Metric, ...] = (
    MAE_METRIC,
    SOFT_METRIC,
    ITERATIONS_METRIC,
    ELAPSED_METRIC,
)


def _short_model_name(model: str) -> str:
    spec = LlmSpec.parse(model)
    short = spec.model.rsplit("/", 1)[-1]
    if spec.reasoning is None:
        return short
    return f"{short} ({spec.reasoning})"


def _bare_short_model_name(model: str) -> str:
    return LlmSpec.parse(model).model.rsplit("/", 1)[-1]


def _ordered(labels: set[str], preferred: list[str]) -> list[str]:
    ordered = [label for label in preferred if label in labels]
    return ordered + sorted(labels - set(ordered))


def _variant_order(variants: set[str]) -> list[str]:
    return _ordered(variants, [variant.label for variant in VARIANTS])


def _model_order(models: set[str]) -> list[str]:
    return _ordered(models, [str(spec) for spec in MODEL_COMPARISON_MODELS])


def _bare_model_order(labels: set[str]) -> list[str]:
    preferred: list[str] = []
    seen: set[str] = set()
    for spec in MODEL_COMPARISON_MODELS:
        name = spec.model.rsplit("/", 1)[-1]
        if name not in seen:
            seen.add(name)
            preferred.append(name)
    return _ordered(labels, preferred)


def _resolved_effort(model: str) -> str:
    spec = LlmSpec.parse(model)
    if spec.reasoning is not None:
        return spec.reasoning
    try:
        return DEFAULT_EFFORT[spec.model]
    except KeyError:
        raise KeyError(f"No default effort for {spec.model}") from None


def _effort_bucket(model: str) -> str:
    effort = _resolved_effort(model)
    try:
        return _EFFORT_TO_BUCKET[effort]
    except KeyError:
        raise ValueError(f"Unknown effort {effort!r} for {model}") from None


def _group_records(records: list[ScoredRun]) -> dict[tuple[str, str], list[ScoredRun]]:
    groups: dict[tuple[str, str], list[ScoredRun]] = {}
    for record in records:
        key = (record.llm_model, record.variant)
        groups.setdefault(key, []).append(record)
    return groups


def _default_grouping(records: list[ScoredRun]) -> Grouping:
    models = {record.llm_model for record in records}
    variants = {record.variant for record in records}

    if len(models) == 1:
        return Grouping(key="", title="", label=lambda record: record.variant, order=_variant_order)

    if len(variants) == 1:
        preferred = [_short_model_name(model) for model in _model_order(models)]
        return Grouping(
            key="",
            title="",
            label=lambda record: _short_model_name(record.llm_model),
            order=lambda labels: _ordered(labels, preferred),
            sort_desc=True,
        )

    pairs = sorted({(record.llm_model, record.variant) for record in records})
    preferred = [f"{_short_model_name(model)} / {variant}" for model, variant in pairs]
    return Grouping(
        key="",
        title="",
        label=lambda record: f"{_short_model_name(record.llm_model)} / {record.variant}",
        order=lambda labels: _ordered(labels, preferred),
        sort_desc=True,
    )


BY_MODEL = Grouping(
    key="by_model",
    title="grouped by model",
    label=lambda record: _bare_short_model_name(record.llm_model),
    order=_bare_model_order,
    sort_desc=True,
)
BY_EFFORT = Grouping(
    key="by_effort",
    title="grouped by effort",
    label=lambda record: _effort_bucket(record.llm_model),
    order=lambda labels: _ordered(labels, list(_EFFORT_BUCKETS)),
)


def grouped_metric_stats(records: list[ScoredRun], metric: Metric, grouping: Grouping) -> list[GroupStats]:
    buckets: dict[str, list[ScoredRun]] = {}
    for record in records:
        buckets.setdefault(grouping.label(record), []).append(record)

    stats: list[GroupStats] = []
    for label in grouping.order(set(buckets)):
        values = [metric.value(record) for record in buckets[label]]
        stats.append(
            GroupStats(
                label=label,
                mean=statistics.fmean(values),
                std=statistics.pstdev(values) if len(values) > 1 else 0.0,
                n=len(values),
            ),
        )
    if grouping.sort_desc:
        stats.sort(key=lambda group: group.mean, reverse=True)
    return stats


def group_metric_stats(records: list[ScoredRun], metric: Metric) -> list[GroupStats]:
    return grouped_metric_stats(records, metric, _default_grouping(records))


def primary_records(records_by_judge: dict[str, list[ScoredRun]]) -> list[ScoredRun]:
    """Records of the first judge that has any, for metrics that do not depend on the judge."""
    for judge in all_judges():
        records = records_by_judge.get(judge.key)
        if records:
            return records
    return []


def group_metric_stats_by_judge(
    records_by_judge: dict[str, list[ScoredRun]],
    metric: Metric,
    grouping: Grouping | None = None,
) -> list[tuple[str, list[GroupStats]]]:
    """Per-judge group stats, every judge aligned to the first judge's group order."""
    primary = primary_records(records_by_judge)
    if not primary:
        return []

    grouping = grouping or _default_grouping(primary)
    label_order = [group.label for group in grouped_metric_stats(primary, metric, grouping)]
    series: list[tuple[str, list[GroupStats]]] = []

    for judge in all_judges():
        records = records_by_judge.get(judge.key)
        if not records:
            continue
        by_label = {group.label: group for group in grouped_metric_stats(records, metric, grouping)}
        aligned = [by_label[label] for label in label_order if label in by_label]
        if aligned:
            series.append((judge.label, aligned))

    return series


def complete_spec_groups(
    pairs: list[tuple[RunSpec, ScoredRun]],
    *,
    reps: int = 3,
) -> list[list[ScoredRun]]:
    groups: dict[tuple[str, str, str], dict[int, ScoredRun]] = {}
    for spec, record in pairs:
        key = (spec.llm.name, spec.variant.label, spec.scenario_id)
        groups.setdefault(key, {})[spec.rep] = record

    result: list[list[ScoredRun]] = []
    for by_rep in groups.values():
        if set(by_rep) != set(range(reps)):
            continue
        result.append([by_rep[rep] for rep in range(reps)])
    return result


def mean_abs_deviation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.fmean([abs(value - mean) for value in values])


def model_variability(groups: list[list[ScoredRun]], metric: Metric) -> list[ModelVariability]:
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


def pooled_variability(groups: list[list[ScoredRun]], metric: Metric) -> ModelVariability:
    spreads = [mean_abs_deviation([metric.value(record) for record in group]) for group in groups]
    return ModelVariability(
        label=POOLED_VARIABILITY_LABEL,
        mean=statistics.fmean(spreads),
        n_specs=len(spreads),
    )


def variability_by_judge(
    groups_by_judge: dict[str, list[list[ScoredRun]]],
    metric: Metric,
) -> tuple[list[str], list[tuple[str, list[ModelVariability | None]]]]:
    all_models: set[str] = set()
    per_judge: dict[str, dict[str, ModelVariability]] = {}
    groups_by_judge_key: dict[str, list[list[ScoredRun]]] = {}

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

    primary_key = next(iter(per_judge))
    model_labels = [_short_model_name(model) for model in _model_order(all_models)]
    model_labels.sort(
        key=lambda label: per_judge[primary_key][label].mean if label in per_judge[primary_key] else 0.0,
        reverse=True,
    )
    label_order = model_labels + [POOLED_VARIABILITY_LABEL]
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
