"""Report rendering: Markdown / LaTeX tables and matplotlib bar charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend for headless reproducibility.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.runner import variant_summary  # noqa: E402

_METRIC_ORDER = ["HSR", "SSR", "CSR", "MAE_pct", "MSE_pct", "Faithfulness", "iterations", "elapsed_s"]


def write_markdown_summary(df: pd.DataFrame, dest: Path) -> None:
    """Write a Markdown summary with one table per metric (variant x level)."""
    summary = variant_summary(df)
    metrics = [m for m in _METRIC_ORDER if m in summary.columns]
    out_lines: list[str] = ["# Ablation study results\n"]

    out_lines.append("## Aggregated mean per (variant, level)\n")
    out_lines.append(summary.to_markdown(index=False))
    out_lines.append("")

    for metric in metrics:
        pivot = summary.pivot(index="variant", columns="level", values=metric)
        out_lines.append(f"\n## {metric} (mean across profiles)\n")
        out_lines.append(pivot.round(3).to_markdown())
    dest.write_text("\n".join(out_lines), encoding="utf-8")


def write_latex_summary(df: pd.DataFrame, dest: Path) -> None:
    """Write a LaTeX `tabular` table ready to be \\input{} into the thesis."""
    summary = variant_summary(df)
    dest.write_text(summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8")


def write_plots(df: pd.DataFrame, dest: Path) -> None:
    """Render a 2x2 matplotlib figure with the four headline metrics."""
    summary = variant_summary(df)
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics_to_plot = [
        ("HSR", "Hard Satisfaction Rate", axes[0][0], False),
        ("CSR", "Constraint Satisfaction Rate", axes[0][1], False),
        ("MAE_pct", "Mean Absolute Macro Error (%)", axes[1][0], True),
        ("Faithfulness", "Faithfulness", axes[1][1], False),
    ]
    for metric, title, ax, lower_is_better in metrics_to_plot:
        if metric not in summary.columns:
            continue
        pivot = summary.pivot(index="variant", columns="level", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(title + (" (lower is better)" if lower_is_better else ""))
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation: V0 -> V4 across patient complexity levels", fontsize=14)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150)
    plt.close(fig)
