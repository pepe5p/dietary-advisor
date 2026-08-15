from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evaluation.judges import all_judges
from evaluation.persistence import resolve_output_dir, save_json
from evaluation.plotting.stats import group_metric_stats, GroupStats, METRICS, primary_records
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
    judges: dict[str, str]
    metrics: dict[str, MetricStats]
    summaries: dict[str, list[VariantSummary]]


def metrics_dir(*, output_dir: Path | None = None) -> Path:
    return resolve_output_dir(output_dir) / "metrics"


def metrics_path(experiment: str, *, output_dir: Path | None = None) -> Path:
    return metrics_dir(output_dir=output_dir) / f"{experiment}.json"


def build_experiment_metrics(
    experiment: str,
    records_by_judge: dict[str, list[ScoreRecord]],
) -> ExperimentMetrics:
    primary = primary_records(records_by_judge)
    metrics: dict[str, MetricStats] = {}
    for metric in METRICS:
        if metric.judge_dependent:
            for judge in all_judges():
                records = records_by_judge.get(judge.key)
                if not records:
                    continue
                metrics[f"{metric.key}_{judge.key}"] = MetricStats(
                    title=f"{metric.title} ({judge.label})",
                    ylabel=metric.ylabel,
                    groups=group_metric_stats(records, metric),
                )
        else:
            metrics[metric.key] = MetricStats(
                title=metric.title,
                ylabel=metric.ylabel,
                groups=group_metric_stats(primary, metric),
            )

    summaries = {
        judge.key: summarize(records_by_judge[judge.key]) for judge in all_judges() if records_by_judge.get(judge.key)
    }
    return ExperimentMetrics(
        experiment=experiment,
        generated_at=datetime.now(timezone.utc),
        n_runs=len(primary),
        judges={judge.key: judge.model_id for judge in all_judges()},
        metrics=metrics,
        summaries=summaries,
    )


def write_experiment_metrics(
    experiment: str,
    records_by_judge: dict[str, list[ScoreRecord]],
    *,
    output_dir: Path | None = None,
) -> Path:
    path = metrics_path(experiment, output_dir=output_dir)
    report = build_experiment_metrics(experiment, records_by_judge)
    return save_json(path, report)
