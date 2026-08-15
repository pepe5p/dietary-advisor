from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.persistence import resolve_output_dir
from evaluation.plotting.stats import (
    _group_records,
    _short_model_name,
    BY_EFFORT,
    BY_MODEL,
    group_metric_stats,
    group_metric_stats_by_judge,
    grouped_metric_stats,
    Grouping,
    GroupStats,
    Metric,
    ModelVariability,
    PLOT_METRICS,
    primary_records,
    SOFT_METRIC,
    SPREAD_COLUMN_LABEL,
    variability_by_judge,
)
from evaluation.scoring.store import ScoreRecord

_DPI = 150
_SERIES_COLORS = ("#4472C4", "#ED7D31", "#70AD47", "#A5A5A5")
_EXTRA_GROUPINGS = (BY_MODEL, BY_EFFORT)


def figures_dir(experiment: str, *, output_dir: Path | None = None) -> Path:
    return resolve_output_dir(output_dir) / "figures" / experiment


def _figure_title(
    experiment: str,
    records: list[ScoreRecord],
    metric: Metric,
    grouping: Grouping | None = None,
) -> str:
    groups = _group_records(records)
    models = {model for model, _ in groups}
    variants = {variant for _, variant in groups}
    extras: list[str] = []
    if len(models) == 1:
        extras.append(_short_model_name(next(iter(models))))
    elif len(variants) == 1:
        extras.append(next(iter(variants)))
    if grouping is not None and grouping.title:
        extras.append(grouping.title)
    if extras:
        return f"{experiment}: {metric.title} ({', '.join(extras)})"
    return f"{experiment}: {metric.title}"


def _filter_records(
    records_by_judge: dict[str, list[ScoreRecord]],
    metric: Metric,
) -> dict[str, list[ScoreRecord]]:
    if metric.applies_to is None:
        return records_by_judge
    return {
        key: [record for record in records if metric.applies_to(record)] for key, records in records_by_judge.items()
    }


def _error_bounds(stats: list[GroupStats], metric: Metric) -> tuple[list[float], list[float]]:
    upper = [group.sem for group in stats]
    if metric.floor is None:
        return upper, upper
    lower = [min(group.sem, max(group.mean - metric.floor, 0.0)) for group in stats]
    return lower, upper


def _series_extents(series: list[tuple[str, list[GroupStats]]], metric: Metric) -> tuple[float, float]:
    low = float("inf")
    high = float("-inf")
    for _, stats in series:
        lower, upper = _error_bounds(stats, metric)
        for group, lo, hi in zip(stats, lower, upper, strict=True):
            low = min(low, group.mean - lo)
            high = max(high, group.mean + hi)
    return low, high


def _apply_axis_limits(ax: plt.Axes, series: list[tuple[str, list[GroupStats]]], metric: Metric) -> None:
    low, high = _series_extents(series, metric)
    if low == float("inf"):
        return
    if metric.zoom_ylim:
        pad = max((high - low) * 0.15, 0.01)
        ax.set_ylim(low - pad, high + pad)
        return
    bottom = metric.floor if metric.floor is not None else low
    pad = max((high - bottom) * 0.1, 0.05)
    ax.set_ylim(bottom, high + pad)


def _render_metric_figure(
    experiment: str,
    records: list[ScoreRecord],
    metric: Metric,
    path: Path,
    *,
    series: list[tuple[str, list]] | None = None,
    grouping: Grouping | None = None,
) -> None:
    if series is None:
        stats = (
            grouped_metric_stats(records, metric, grouping)
            if grouping is not None
            else group_metric_stats(records, metric)
        )
        series = [("", stats)]

    labels = [group.label for group in series[0][1]]
    n_groups = len(labels)
    n_series = len(series)
    bar_width = 0.8 / n_series
    show_error_bars = any(group.n > 1 for _, stats in series for group in stats)

    fig, ax = plt.subplots(figsize=(max(7.0, n_groups * 1.4), 4.5))
    x = range(n_groups)

    for series_idx, (series_label, stats) in enumerate(series):
        means = [group.mean for group in stats]
        lower, upper = _error_bounds(stats, metric)
        offsets = [idx + (series_idx - (n_series - 1) / 2) * bar_width for idx in x]
        color = _SERIES_COLORS[series_idx % len(_SERIES_COLORS)]
        ax.bar(
            offsets,
            means,
            width=bar_width,
            yerr=[lower, upper] if show_error_bars else None,
            capsize=4 if show_error_bars else 0,
            color=color,
            label=series_label or None,
        )
        for offset, mean, hi in zip(offsets, means, upper, strict=True):
            label_y = mean + hi if show_error_bars else mean
            ax.text(
                offset,
                label_y,
                f"{mean:.{metric.decimals}f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.ylabel)
    title = textwrap.fill(_figure_title(experiment, records, metric, grouping), width=60)
    ax.set_title(title, fontsize=10, pad=14)
    ax.text(1.0, 1.02, "error bars: +/- SEM", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if n_series > 1:
        ax.legend(fontsize=8)
    _apply_axis_limits(ax, series, metric)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_metric_figure(
    name: str,
    records_by_judge: dict[str, list[ScoreRecord]],
    metric: Metric,
    path: Path,
    grouping: Grouping | None = None,
) -> list[Path]:
    primary = primary_records(records_by_judge)
    if metric.judge_dependent:
        series = group_metric_stats_by_judge(records_by_judge, metric, grouping=grouping)
        _render_metric_figure(name, primary, metric, path, series=series, grouping=grouping)
    else:
        _render_metric_figure(name, primary, metric, path, grouping=grouping)
    return [path, path.with_suffix(".pdf")]


def render_experiment(
    name: str,
    records_by_judge: dict[str, list[ScoreRecord]],
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    primary = primary_records(records_by_judge)
    if not primary:
        return []

    extra_groupings = _EXTRA_GROUPINGS if len({record.llm_model for record in primary}) > 1 else ()
    dest = figures_dir(name, output_dir=output_dir)
    paths: list[Path] = []
    for metric in PLOT_METRICS:
        filtered = _filter_records(records_by_judge, metric)
        if not primary_records(filtered):
            continue
        paths.extend(_write_metric_figure(name, filtered, metric, dest / metric.filename))
        for grouping in extra_groupings:
            extra_path = dest / f"{metric.key}_{grouping.key}.png"
            paths.extend(_write_metric_figure(name, filtered, metric, extra_path, grouping=grouping))
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
    label_order, series = variability_by_judge(groups_by_judge, SOFT_METRIC)
    if not series:
        return []

    dest = figures_dir("stability", output_dir=output_dir)
    path = dest / "run_variability.png"

    fig, ax = plt.subplots(figsize=(max(8.0, len(label_order) * 1.2), 4.5))
    _render_variability_subplot(ax, label_order, series, SOFT_METRIC)

    fig.suptitle(f"Run-to-run variability ({reps} reps per spec)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [path, path.with_suffix(".pdf")]
