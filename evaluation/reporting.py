from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evaluation.persistence import resolve_output_dir, save_json
from evaluation.plotting.stats import group_metric_stats, GroupStats, METRICS
from evaluation.scoring.aggregate import summarize, VariantSummary
from evaluation.scoring.store import ScoreRecord


class MetricStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    ylabel: str
    groups: list[GroupStats]


class ExperimentMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment: str
    generated_at: datetime
    n_runs: int
    metrics: dict[str, MetricStats]
    summaries: list[VariantSummary]


def metrics_dir(*, output_dir: Path | None = None) -> Path:
    return resolve_output_dir(output_dir) / "metrics"


def metrics_path(experiment: str, *, output_dir: Path | None = None) -> Path:
    return metrics_dir(output_dir=output_dir) / f"{experiment}.json"


def build_experiment_metrics(experiment: str, records: list[ScoreRecord]) -> ExperimentMetrics:
    metrics: dict[str, MetricStats] = {}
    for metric in METRICS:
        metrics[metric.key] = MetricStats(
            title=metric.title,
            ylabel=metric.ylabel,
            groups=group_metric_stats(records, metric),
        )
    return ExperimentMetrics(
        experiment=experiment,
        generated_at=datetime.now(timezone.utc),
        n_runs=len(records),
        metrics=metrics,
        summaries=summarize(records),
    )


def write_experiment_metrics(
    experiment: str,
    records: list[ScoreRecord],
    *,
    output_dir: Path | None = None,
) -> Path:
    path = metrics_path(experiment, output_dir=output_dir)
    report = build_experiment_metrics(experiment, records)
    return save_json(path, report)
