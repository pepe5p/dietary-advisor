from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.judges import all_judges
from evaluation.persistence import resolve_output_dir
from evaluation.plotting.stats import (
    _group_records,
    _short_model_name,
    group_metric_stats,
    group_metric_stats_by_judge,
    Metric,
    METRICS,
    ModelVariability,
    primary_records,
    SPREAD_COLUMN_LABEL,
    variability_by_judge,
)
from evaluation.scoring.store import ScoreRecord

_DPI = 150
_SERIES_COLORS = ("#4472C4", "#ED7D31", "#70AD47", "#A5A5A5")
_SOFT_VARIABILITY_METRIC = METRICS[1]


def figures_dir(experiment: str, *, output_dir: Path | None = None) -> Path:
    return resolve_output_dir(output_dir) / "figures" / experiment


def _figure_title(experiment: str, records: list[ScoreRecord], metric: Metric) -> str:
    groups = _group_records(records)
    models = {model for model, _ in groups}
    variants = {variant for _, variant in groups}
    if len(models) == 1:
        model = _short_model_name(next(iter(models)))
        return f"{experiment}: {metric.title} ({model})"
    if len(variants) == 1:
        return f"{experiment}: {metric.title} ({next(iter(variants))})"
    return f"{experiment}: {metric.title}"


def _render_metric_figure(
    experiment: str,
    records: list[ScoreRecord],
    metric: Metric,
    path: Path,
    *,
    series: list[tuple[str, list]] | None = None,
) -> None:
    if series is None:
        stats = group_metric_stats(records, metric)
        series = [("", stats)]

    labels = [group.label for group in series[0][1]]
    n_groups = len(labels)
    n_series = len(series)
    bar_width = 0.8 / n_series
    show_error_bars = any(group.n > 1 for _, stats in series for group in stats)

    fig, ax = plt.subplots(figsize=(max(6.0, n_groups * 1.4), 4.5))
    x = range(n_groups)

    for series_idx, (series_label, stats) in enumerate(series):
        means = [group.mean for group in stats]
        stds = [group.std for group in stats]
        offsets = [idx + (series_idx - (n_series - 1) / 2) * bar_width for idx in x]
        color = _SERIES_COLORS[series_idx % len(_SERIES_COLORS)]
        ax.bar(
            offsets,
            means,
            width=bar_width,
            yerr=stds if show_error_bars else None,
            capsize=4 if show_error_bars else 0,
            color=color,
            label=series_label or None,
        )
        for offset, mean in zip(offsets, means, strict=True):
            ax.text(offset, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.ylabel)
    run_counts = ", ".join(str(group.n) for group in series[0][1])
    ax.set_title(f"{_figure_title(experiment, records, metric)}\n(n={run_counts})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if n_series > 1:
        ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def render_experiment(
    name: str,
    records_by_judge: dict[str, list[ScoreRecord]],
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    primary = primary_records(records_by_judge)
    if not primary:
        return []

    dest = figures_dir(name, output_dir=output_dir)
    paths: list[Path] = []
    for metric in METRICS:
        path = dest / metric.filename
        if metric.judge_dependent:
            series = group_metric_stats_by_judge(records_by_judge, metric)
            _render_metric_figure(name, primary, metric, path, series=series)
        else:
            _render_metric_figure(name, primary, metric, path)
        paths.append(path)
        paths.append(path.with_suffix(".pdf"))
    return paths


def _render_variability_subplot(
    ax: plt.Axes,
    label_order: list[str],
    series: list[tuple[str, list[ModelVariability | None]]],
    metric: Metric,
) -> None:
    n_groups = len(label_order)
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = range(n_groups)

    for series_idx, (series_label, stats) in enumerate(series):
        offsets = [idx + (series_idx - (n_series - 1) / 2) * bar_width for idx in x]
        color = _SERIES_COLORS[series_idx % len(_SERIES_COLORS)]
        ax.bar(
            offsets,
            [entry.mean if entry else float("nan") for entry in stats],
            width=bar_width,
            color=color,
            label=series_label,
        )
        for offset, entry in zip(offsets, stats, strict=True):
            if entry is None:
                continue
            ax.text(offset, entry.mean, f"{entry.mean:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(label_order, rotation=20, ha="right")
    ax.set_ylabel(SPREAD_COLUMN_LABEL)
    ax.set_title(metric.title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if n_series > 1:
        ax.legend(fontsize=8)


def render_run_variability(
    groups_by_judge: dict[str, list[list[ScoreRecord]]],
    *,
    output_dir: Path | None = None,
    reps: int = 3,
) -> list[Path]:
    scored = [(judge, groups_by_judge[judge.key]) for judge in all_judges() if groups_by_judge.get(judge.key)]
    if not scored:
        return []

    label_order, series = variability_by_judge(groups_by_judge, _SOFT_VARIABILITY_METRIC)
    if not series:
        return []

    dest = figures_dir("stability", output_dir=output_dir)
    path = dest / "run_variability.png"

    fig, ax = plt.subplots(figsize=(max(8.0, len(label_order) * 1.2), 4.5))
    _render_variability_subplot(ax, label_order, series, _SOFT_VARIABILITY_METRIC)

    spec_counts = ", ".join(f"{judge.label} {len(groups)}" for judge, groups in scored)
    fig.suptitle(f"Run-to-run variability ({reps} reps per spec; complete specs: {spec_counts})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [path, path.with_suffix(".pdf")]
