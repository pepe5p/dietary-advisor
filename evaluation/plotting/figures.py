from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.persistence import resolve_output_dir
from evaluation.plotting.stats import (
    _group_records,
    _short_model_name,
    group_metric_stats,
    Metric,
    METRICS,
    scenario_spread_stats,
    sem_for_reps,
    SpreadStats,
)
from evaluation.scoring.store import ScoreRecord

_DPI = 150


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
) -> None:
    stats = group_metric_stats(records, metric)
    labels = [group.label for group in stats]
    means = [group.mean for group in stats]
    stds = [group.std for group in stats]
    show_error_bars = any(group.n > 1 for group in stats)

    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 1.4), 4.5))
    x = range(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds if show_error_bars else None,
        capsize=4 if show_error_bars else 0,
        color="#4472C4",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.ylabel)
    run_counts = ", ".join(str(group.n) for group in stats)
    ax.set_title(f"{_figure_title(experiment, records, metric)}\n(n={run_counts})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for idx, mean in enumerate(means):
        ax.text(idx, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    # Vector copy for the thesis: LaTeX floats should not embed raster bar charts.
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def render_experiment(
    name: str,
    records: list[ScoreRecord],
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    dest = figures_dir(name, output_dir=output_dir)
    paths: list[Path] = []
    for metric in METRICS:
        path = dest / metric.filename
        _render_metric_figure(name, records, metric, path)
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
    ax.set_title("Deviation from spec mean")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _render_spread_magnitude(
    ax: plt.Axes,
    per_scenario: list[SpreadStats],
    pooled: SpreadStats,
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
    ax.set_title("Spread magnitude")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _render_sem_curve(
    ax: plt.Axes,
    per_scenario: list[SpreadStats],
    pooled: SpreadStats,
    *,
    max_reps: int,
    current_reps: int,
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
    ax.set_title("Precision vs repetitions")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")


def render_run_variability(
    groups: list[list[ScoreRecord]],
    *,
    output_dir: Path | None = None,
    max_reps: int = 10,
    reps: int = 3,
) -> list[Path]:
    dest = figures_dir("stability", output_dir=output_dir)
    path = dest / "run_variability.png"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, metric in enumerate((METRICS[0], METRICS[1])):
        per_scenario, pooled = _split_scenario_stats(scenario_spread_stats(groups, metric))
        _render_deviation_violins(axes[row, 0], per_scenario, pooled, metric)
        _render_spread_magnitude(axes[row, 1], per_scenario, pooled)
        _render_sem_curve(axes[row, 2], per_scenario, pooled, max_reps=max_reps, current_reps=reps)

    fig.suptitle(f"Run-to-run variability ({len(groups)} specs x {reps} reps)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [path, path.with_suffix(".pdf")]
