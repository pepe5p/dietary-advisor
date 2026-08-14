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
