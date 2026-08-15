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
    primary_records,
    scenario_spread_stats,
    sem_for_reps,
    SpreadStats,
)
from evaluation.scoring.store import ScoreRecord

_DPI = 150
_SERIES_COLORS = ("#4472C4", "#ED7D31", "#70AD47", "#A5A5A5")


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


def _split_scenario_stats(stats: list[SpreadStats]) -> tuple[list[SpreadStats], SpreadStats]:
    per_scenario = [entry for entry in stats if entry.label != "all"]
    pooled = next(entry for entry in stats if entry.label == "all")
    return per_scenario, pooled


def _render_deviation_violins(
    ax: plt.Axes,
    per_scenario: list[SpreadStats],
    pooled: SpreadStats,
    metric: Metric,
    *,
    title_suffix: str = "",
) -> None:
    entries = per_scenario + [pooled]
    labels = [entry.label for entry in entries]
    positions = list(range(len(entries)))

    ax.violinplot(
        [entry.deviations for entry in entries],
        positions=positions,
        showmeans=False,
        showextrema=False,
    )
    ax.boxplot(
        [entry.deviations for entry in entries],
        positions=positions,
        widths=0.15,
        patch_artist=True,
        boxprops={"facecolor": "white", "alpha": 0.7},
        medianprops={"color": "black"},
        whiskerprops={"linewidth": 1},
        capprops={"linewidth": 1},
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.ylabel)
    title = "Deviation from spec mean"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _render_spread_magnitude(
    ax: plt.Axes,
    per_scenario: list[SpreadStats],
    pooled: SpreadStats,
    *,
    title_suffix: str = "",
) -> None:
    entries = per_scenario + [pooled]
    labels = [entry.label for entry in entries]
    positions = list(range(len(entries)))
    pooled_stds = [entry.pooled_std for entry in entries]

    ax.bar(positions, pooled_stds, color="#4472C4")
    for idx, entry in enumerate(entries):
        if not entry.per_spec_std:
            continue
        spread = 0.24
        denominator = max(len(entry.per_spec_std) - 1, 1)
        offsets = [
            idx + (position - (len(entry.per_spec_std) - 1) / 2) * spread / denominator
            for position in range(len(entry.per_spec_std))
        ]
        ax.scatter(offsets, entry.per_spec_std, color="#203864", s=18, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Within-spec std")
    title = "Spread magnitude"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _render_sem_curve(
    ax: plt.Axes,
    per_scenario: list[SpreadStats],
    pooled: SpreadStats,
    *,
    max_reps: int,
    current_reps: int,
    title_suffix: str = "",
) -> None:
    rep_counts = list(range(1, max_reps + 1))
    for entry in per_scenario:
        sems = [sem_for_reps(entry.pooled_std, n) for n in rep_counts]
        ax.plot(rep_counts, sems, marker="o", markersize=4, label=entry.label)

    pooled_sems = [sem_for_reps(pooled.pooled_std, n) for n in rep_counts]
    ax.plot(rep_counts, pooled_sems, marker="o", markersize=4, linestyle="--", label="all")

    ax.axvline(current_reps, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Repetitions")
    ax.set_ylabel("SEM of spec mean")
    title = "Precision vs repetitions"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")


def render_run_variability(
    groups_by_judge: dict[str, list[list[ScoreRecord]]],
    *,
    output_dir: Path | None = None,
    max_reps: int = 10,
    reps: int = 3,
) -> list[Path]:
    scored = [(judge, groups_by_judge[judge.key]) for judge in all_judges() if groups_by_judge.get(judge.key)]
    if not scored:
        return []

    dest = figures_dir("stability", output_dir=output_dir)
    path = dest / "run_variability.png"

    primary_judge, primary_groups = scored[0]
    rows: list[tuple[Metric, list[list[ScoreRecord]], str]] = [
        (METRICS[0], primary_groups, primary_judge.label),
        *((METRICS[1], groups, judge.label) for judge, groups in scored),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4 * len(rows)), squeeze=False)
    for row_idx, (metric, groups, suffix) in enumerate(rows):
        per_scenario, pooled = _split_scenario_stats(scenario_spread_stats(groups, metric))
        _render_deviation_violins(axes[row_idx, 0], per_scenario, pooled, metric, title_suffix=suffix)
        _render_spread_magnitude(axes[row_idx, 1], per_scenario, pooled, title_suffix=suffix)
        _render_sem_curve(
            axes[row_idx, 2],
            per_scenario,
            pooled,
            max_reps=max_reps,
            current_reps=reps,
            title_suffix=suffix,
        )

    fig.suptitle(f"Run-to-run variability ({len(primary_groups)} specs x {reps} reps)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [path, path.with_suffix(".pdf")]
