"""Render scored evaluation runs as booktabs LaTeX tables for the thesis.

Each function returns bare ``tabular`` content (no ``table`` float, caption or
label) so the thesis chapters control placement, captions and labels; only the
numbers are generated here.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from evaluation.case_runner.grid import experiment_runs, EXPERIMENTS
from evaluation.judges import all_judges
from evaluation.plotting.stats import _short_model_name, _variant_order, primary_records
from evaluation.records import scored_runs, ScoredRun

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TABLES_DIR = _REPO_ROOT / "thesis" / "tables"


def _escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _scored_records_by_judge(experiment: str) -> dict[str, list[ScoredRun]]:
    specs = experiment_runs(experiment)
    return {judge.key: [record for _, record in scored_runs(specs, judge=judge)] for judge in all_judges()}


def _group_by(records: list[ScoredRun], key: str) -> dict[str, list[ScoredRun]]:
    groups: dict[str, list[ScoredRun]] = {}
    for record in records:
        groups.setdefault(getattr(record, key), []).append(record)
    return groups


# Trailing backslash is LaTeX's escaped inter-word space, keeping "agg." from being read as a sentence end.
_SOFT_METRIC = ("Soft agg.\\", "aggregate")
_SAFETY_METRIC = ("Safety", "safety_adherence")


def _judge_means(records: list[ScoredRun], key: str, attribute: str) -> dict[str, float]:
    groups = _group_by(records, key)
    return {
        label: statistics.fmean(getattr(r.qualitative, attribute) for r in group) for label, group in groups.items()
    }


def _judge_columns(
    records_by_judge: dict[str, list[ScoredRun]],
    key: str,
    metrics: tuple[tuple[str, str], ...],
) -> list[tuple[str, dict[str, float]]]:
    """One (header, per-group mean) column per judge and metric, judges in registry order."""
    return [
        (f"{header} {_escape(judge.label)}", _judge_means(records_by_judge.get(judge.key, []), key, attribute))
        for judge in all_judges()
        for header, attribute in metrics
    ]


def _judge_cells(columns: list[tuple[str, dict[str, float]]], group_key: str) -> list[str]:
    cells = []
    for _, means in columns:
        value = means.get(group_key)
        cells.append(f"{value:.3f}" if value is not None else "--")
    return cells


def variant_summary_table(records_by_judge: dict[str, list[ScoredRun]]) -> str:
    """One row per ablation variant: error, preference and operational metrics, averaged over all runs."""
    groups = _group_by(primary_records(records_by_judge), "variant")
    order = _variant_order(set(groups))
    judge_columns = _judge_columns(records_by_judge, "variant", (_SOFT_METRIC, _SAFETY_METRIC))

    headers = ["Variant", r"MAE~[\%]", r"MSE~[\%]", "Iter.", r"Elapsed~[s]"]
    headers += [head for head, _ in judge_columns]
    lines = [
        rf"\begin{{tabular}}{{{'l' + 'r' * (len(headers) - 1)}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for label in order:
        group = groups[label]
        cells = [
            _escape(label),
            f"{statistics.fmean(r.mae_pct for r in group):.2f}",
            f"{statistics.fmean(r.mse_pct for r in group):.2f}",
            f"{statistics.fmean(r.iterations for r in group):.2f}",
            f"{statistics.fmean(r.elapsed_s for r in group):.1f}",
            *_judge_cells(judge_columns, label),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def model_summary_table(records_by_judge: dict[str, list[ScoredRun]]) -> str:
    """One row per model under the full variant: error, preference and operational metrics."""
    groups = _group_by(primary_records(records_by_judge), "llm_model")
    soft_columns = _judge_columns(records_by_judge, "llm_model", (_SOFT_METRIC,))

    headers = ["Model", r"MAE~[\%]", *(head for head, _ in soft_columns), "Iter.", r"Elapsed~[s]"]
    lines = [
        rf"\begin{{tabular}}{{{'l' + 'r' * (len(headers) - 1)}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for model in sorted(groups):
        group = groups[model]
        cells = [
            _escape(_short_model_name(model)),
            f"{statistics.fmean(r.mae_pct for r in group):.2f}",
            *_judge_cells(soft_columns, model),
            f"{statistics.fmean(r.iterations for r in group):.2f}",
            f"{statistics.fmean(r.elapsed_s for r in group):.1f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_scenario_mae_table(records: list[ScoredRun]) -> str:
    """Mean MAE% for every (scenario, variant) pair of a single-model ablation grid."""
    variants = _variant_order({r.variant for r in records})
    scenarios = sorted({r.scenario_id for r in records})

    by_pair: dict[tuple[str, str], list[float]] = {}
    for record in records:
        by_pair.setdefault((record.scenario_id, record.variant), []).append(record.mae_pct)

    col_spec = "l" + "r" * len(variants)
    header = "Scenario & " + " & ".join(_escape(v) for v in variants) + r" \\"
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for scenario in scenarios:
        cells = []
        for variant in variants:
            values = by_pair.get((scenario, variant), [])
            cells.append(f"{statistics.fmean(values):.1f}" if values else "--")
        lines.append(f"{_escape(scenario)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_tables(*, tables_dir: Path | None = None) -> list[Path]:
    """Regenerate every thesis table from the currently scored runs. Returns the written paths."""
    dest = tables_dir if tables_dir is not None else _TABLES_DIR
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if "ablation" in EXPERIMENTS:
        ablation_scores = _scored_records_by_judge("ablation")
        ablation_primary = primary_records(ablation_scores)
        if ablation_primary:
            for name, content in (
                ("ablation_variant_summary.tex", variant_summary_table(ablation_scores)),
                ("ablation_per_scenario_mae.tex", per_scenario_mae_table(ablation_primary)),
            ):
                path = dest / name
                path.write_text(content + "\n", encoding="utf-8")
                written.append(path)

    if "models" in EXPERIMENTS:
        model_scores = _scored_records_by_judge("models")
        if primary_records(model_scores):
            path = dest / "model_comparison_summary.tex"
            path.write_text(model_summary_table(model_scores) + "\n", encoding="utf-8")
            written.append(path)

    return written
