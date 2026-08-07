from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from evaluation.case_runner.grid import MODEL_COMPARISON_MODELS, VARIANTS
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
    if "/" in model:
        return model.rsplit("/", 1)[-1]
    return model


def _variant_order(variants: set[str]) -> list[str]:
    preferred = [variant.label for variant in VARIANTS]
    ordered = [label for label in preferred if label in variants]
    return ordered + sorted(variants - set(ordered))


def _model_order(models: set[str]) -> list[str]:
    ordered = [model for model in MODEL_COMPARISON_MODELS if model in models]
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
