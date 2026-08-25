from __future__ import annotations

import math
import statistics
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from evaluation.judges import all_judges
from evaluation.persistence import resolve_output_dir
from evaluation.plotting.stats import (
    _bare_model_order,
    _bare_short_model_name,
    _default_grouping,
    _group_records,
    _resolved_effort,
    _short_model_name,
    BY_EFFORT,
    BY_MODEL,
    group_metric_stats,
    group_metric_stats_by_judge,
    grouped_metric_stats,
    Grouping,
    GroupStats,
    MAE_METRIC,
    Metric,
    ModelVariability,
    PLOT_METRICS,
    primary_records,
    SOFT_METRIC,
    SPREAD_COLUMN_LABEL,
    variability_by_judge,
)
from evaluation.records import ScoredRun

_DPI = 150
_SERIES_COLORS = ("#4472C4", "#ED7D31", "#70AD47", "#A5A5A5")
_EXTRA_GROUPINGS = (BY_MODEL, BY_EFFORT)
_TRADEOFF_HEIGHT_IN = 4.6
_X_TICK_CANDIDATES = (0.5, 1, 2, 3, 5, 7, 10, 20, 30, 50, 100)
# Scatter series carry a shape as well as a colour, so they stay distinguishable in greyscale.
_MARKERS = ("o", "s", "^", "D", "v", "P")

# A4 with the aghdpl margins (30mm left, 20mm right) leaves a 160mm text block.
# Every figure is drawn exactly that wide so the thesis can include it at
# \textwidth unscaled, which makes the point sizes below the sizes LaTeX prints
# next to 11pt body text.
_TEXT_WIDTH_IN = 160 / 25.4
_AXES_HEIGHT_IN = 2.8
_MIN_HEIGHT_IN = 3.2
# Room taken by the y-axis label plus its tick labels, excluded from the slot
# arithmetic that decides whether a label fits.
_Y_AXIS_ALLOWANCE_IN = 1.0
_SLANT_DEGREES = 30
_TITLE_SIZE = 10
_AXIS_LABEL_SIZE = 9
_TICK_SIZE = 8
_VALUE_SIZE = 7
_LEGEND_SIZE = 8
_NOTE_SIZE = 7
# Mean glyph advance of a mixed-case label, per point of font size.
_CHAR_IN_PER_PT = 0.0068


def _text_len_in(text: str, fontsize: int) -> float:
    return len(text) * fontsize * _CHAR_IN_PER_PT


@dataclass(frozen=True)
class _Layout:
    figsize: tuple[float, float]
    tick_rotation: int
    tick_alignment: str
    value_rotation: int
    headroom: float
    title_wrap: int


def _plan_layout(labels: list[str], value_texts: list[str], n_series: int) -> _Layout:
    """Pick rotations and a figure height that keep every label legible at print size."""
    axes_width = _TEXT_WIDTH_IN - _Y_AXIS_ALLOWANCE_IN
    slot = axes_width / max(len(labels), 1)

    widest_tick = max((_text_len_in(label, _TICK_SIZE) for label in labels), default=0.0)
    slanted = widest_tick * math.cos(math.radians(_SLANT_DEGREES)) <= slot
    tick_block = widest_tick * (math.sin(math.radians(_SLANT_DEGREES)) if slanted else 1.0)

    widest_value = max((_text_len_in(text, _VALUE_SIZE) for text in value_texts), default=0.0)
    upright_values = widest_value <= slot / n_series
    headroom = 0.03 if upright_values else (widest_value + 0.06) / _AXES_HEIGHT_IN

    return _Layout(
        figsize=(_TEXT_WIDTH_IN, max(_MIN_HEIGHT_IN, _AXES_HEIGHT_IN + tick_block + 0.15)),
        tick_rotation=_SLANT_DEGREES if slanted else 90,
        tick_alignment="right" if slanted else "center",
        value_rotation=0 if upright_values else 90,
        headroom=headroom,
        title_wrap=int(_TEXT_WIDTH_IN / (_TITLE_SIZE * _CHAR_IN_PER_PT)),
    )


def figures_dir(experiment: str, *, output_dir: Path | None = None) -> Path:
    return resolve_output_dir(output_dir) / "figures" / experiment


def _figure_title(
    experiment: str,
    records: list[ScoredRun],
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
    records_by_judge: dict[str, list[ScoredRun]],
    metric: Metric,
) -> dict[str, list[ScoredRun]]:
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


def _apply_axis_limits(
    ax: plt.Axes,
    series: list[tuple[str, list[GroupStats]]],
    metric: Metric,
    headroom: float,
) -> None:
    low, high = _series_extents(series, metric)
    if low == float("inf"):
        return
    if metric.zoom_ylim:
        pad = max((high - low) * 0.15, 0.01)
        bottom, top = low - pad, high + pad
    else:
        bottom = metric.floor if metric.floor is not None else low
        top = high + max((high - bottom) * 0.1, 0.05)
    ax.set_ylim(bottom, top + (top - bottom) * headroom)


def _render_metric_figure(
    experiment: str,
    records: list[ScoredRun],
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
    value_texts = [f"{group.mean:.{metric.decimals}f}" for _, stats in series for group in stats]
    layout = _plan_layout(labels, value_texts, n_series)

    fig, ax = plt.subplots(figsize=layout.figsize)
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
            capsize=3 if show_error_bars else 0,
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
                fontsize=_VALUE_SIZE,
                rotation=layout.value_rotation,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=layout.tick_rotation, ha=layout.tick_alignment)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.set_ylabel(metric.ylabel, fontsize=_AXIS_LABEL_SIZE)
    title = textwrap.fill(_figure_title(experiment, records, metric, grouping), width=layout.title_wrap)
    ax.set_title(title, fontsize=_TITLE_SIZE, pad=16)
    if show_error_bars:
        ax.text(
            0.5,
            1.0,
            "error bars: \u00b1SEM",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=_NOTE_SIZE,
        )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if n_series > 1:
        ax.legend(fontsize=_LEGEND_SIZE)
    _apply_axis_limits(ax, series, metric, layout.headroom)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_metric_figure(
    name: str,
    records_by_judge: dict[str, list[ScoredRun]],
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


@dataclass(frozen=True)
class _TradeoffPoint:
    label: str
    model: str
    effort: str
    mae: float
    soft: float


def _tradeoff_points(records: list[ScoredRun]) -> list[_TradeoffPoint]:
    """Join MAE and soft-aggregate stats by group label; the two metrics sort independently."""
    grouping = _default_grouping(records)
    mae_by_label = {group.label: group for group in grouped_metric_stats(records, MAE_METRIC, grouping)}
    soft_by_label = {group.label: group for group in grouped_metric_stats(records, SOFT_METRIC, grouping)}
    sample: dict[str, ScoredRun] = {}
    for record in records:
        sample.setdefault(grouping.label(record), record)

    points: list[_TradeoffPoint] = []
    for label, mae in mae_by_label.items():
        if label not in soft_by_label or label not in sample:
            continue
        record = sample[label]
        points.append(
            _TradeoffPoint(
                label=label,
                model=_bare_short_model_name(record.llm_model),
                effort=_resolved_effort(record.llm_model),
                mae=mae.mean,
                soft=soft_by_label[label].mean,
            ),
        )
    return points


def _mean_tradeoff_points(points_by_judge: list[list[_TradeoffPoint]]) -> list[_TradeoffPoint]:
    """Per-configuration soft aggregate averaged over the judges; MAE is judge-independent."""
    soft_by_label: dict[str, list[float]] = {}
    for points in points_by_judge:
        for point in points:
            soft_by_label.setdefault(point.label, []).append(point.soft)

    shared = [point for point in points_by_judge[0] if len(soft_by_label[point.label]) == len(points_by_judge)]
    return [replace(point, soft=statistics.fmean(soft_by_label[point.label])) for point in shared]


def _render_tradeoff_figure(
    experiment: str,
    judge_label: str,
    points: list[_TradeoffPoint],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(_TEXT_WIDTH_IN, _TRADEOFF_HEIGHT_IN))
    models = _bare_model_order({point.model for point in points})

    for idx, model in enumerate(models):
        group = [point for point in points if point.model == model]
        ax.scatter(
            [point.mae for point in group],
            [point.soft for point in group],
            color=_SERIES_COLORS[idx % len(_SERIES_COLORS)],
            marker=_MARKERS[idx % len(_MARKERS)],
            s=30,
            zorder=3,
            label=model,
        )
    for point in points:
        ax.annotate(
            point.effort,
            (point.mae, point.soft),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=_VALUE_SIZE,
        )

    ax.set_xscale("log")
    xs = [point.mae for point in points]
    x_low, x_high = min(xs) / 1.3, max(xs) * 1.3
    ax.set_xlim(x_low, x_high)
    # Matplotlib's own log ticks mix decade and subdecade labels at two different
    # sizes and collide once the span is under two decades, so place them by hand.
    ax.xaxis.set_major_locator(FixedLocator([tick for tick in _X_TICK_CANDIDATES if x_low <= tick <= x_high]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel(f"{MAE_METRIC.ylabel} ($\\log_{{10}}$ scale)", fontsize=_AXIS_LABEL_SIZE)
    ax.set_ylabel(SOFT_METRIC.title, fontsize=_AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    title = textwrap.fill(
        f"{experiment}: MAE vs soft preference ({judge_label})",
        width=int(_TEXT_WIDTH_IN / (_TITLE_SIZE * _CHAR_IN_PER_PT)),
    )
    ax.set_title(title, fontsize=_TITLE_SIZE, pad=16)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=_LEGEND_SIZE, loc="best")

    ys = [point.soft for point in points]
    y_low, y_high = min(ys), max(ys)
    pad = max((y_high - y_low) * 0.15, 0.01)
    ax.set_ylim(y_low - pad, y_high + pad)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def render_tradeoff(
    experiment: str,
    records_by_judge: dict[str, list[ScoredRun]],
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    dest = figures_dir(experiment, output_dir=output_dir)
    paths: list[Path] = []
    per_judge: list[list[_TradeoffPoint]] = []
    judge_labels: list[str] = []
    for judge in all_judges():
        records = records_by_judge.get(judge.key)
        if not records:
            continue
        points = _tradeoff_points(records)
        if not points:
            continue
        per_judge.append(points)
        judge_labels.append(judge.label)
        path = dest / f"tradeoff_{judge.key}.png"
        _render_tradeoff_figure(experiment, judge.label, points, path)
        paths.extend([path, path.with_suffix(".pdf")])

    if len(per_judge) > 1:
        mean_points = _mean_tradeoff_points(per_judge)
        if mean_points:
            path = dest / "tradeoff_judge_mean.png"
            _render_tradeoff_figure(experiment, f"mean of {' and '.join(judge_labels)}", mean_points, path)
            paths.extend([path, path.with_suffix(".pdf")])
    return paths


def render_experiment(
    name: str,
    records_by_judge: dict[str, list[ScoredRun]],
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
    if extra_groupings:
        paths.extend(render_tradeoff(name, records_by_judge, output_dir=output_dir))
    return paths


def _render_variability_subplot(
    ax: plt.Axes,
    label_order: list[str],
    series: list[tuple[str, list[ModelVariability | None]]],
    layout: _Layout,
) -> None:
    n_groups = len(label_order)
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = range(n_groups)
    high = 0.0

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
            high = max(high, entry.mean)
            ax.text(
                offset,
                entry.mean,
                f"{entry.mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=_VALUE_SIZE,
                rotation=layout.value_rotation,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(label_order, rotation=layout.tick_rotation, ha=layout.tick_alignment)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.set_ylabel(SPREAD_COLUMN_LABEL, fontsize=_AXIS_LABEL_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if n_series > 1:
        ax.legend(fontsize=_LEGEND_SIZE)
    ax.set_ylim(0.0, high * (1.1 + layout.headroom))


def render_run_variability(
    groups_by_judge: dict[str, list[list[ScoredRun]]],
    *,
    output_dir: Path | None = None,
    reps: int = 3,
) -> list[Path]:
    label_order, series = variability_by_judge(groups_by_judge, SOFT_METRIC)
    if not series:
        return []

    dest = figures_dir("stability", output_dir=output_dir)
    path = dest / "run_variability.png"

    value_texts = [f"{entry.mean:.3f}" for _, stats in series for entry in stats if entry]
    layout = _plan_layout(label_order, value_texts, len(series))

    fig, ax = plt.subplots(figsize=layout.figsize)
    _render_variability_subplot(ax, label_order, series, layout)

    title = f"Run-to-run variability: {SOFT_METRIC.title} ({reps} reps per spec)"
    ax.set_title(textwrap.fill(title, width=layout.title_wrap), fontsize=_TITLE_SIZE)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [path, path.with_suffix(".pdf")]
