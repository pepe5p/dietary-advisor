"""Render scored evaluation runs as booktabs LaTeX tables for the thesis.

Each function returns bare ``tabular`` content (no ``table`` float, caption or
label) so the thesis chapters control placement, captions and labels; only the
numbers are generated here.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from pathlib import Path

from dietary_advisor.config.llm import LlmSpec
from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.case_runner.grid import BASELINE, experiment_runs, EXPERIMENTS, FULL_VARIANT
from evaluation.judges import all_judges
from evaluation.paired import paired_sign_test, SignTest
from evaluation.plotting.stats import (
    _model_order,
    _resolved_effort,
    _variant_order,
    MAE_METRIC,
    Metric,
    primary_records,
    SAFETY_METRIC,
    SOFT_METRIC,
    variant_display_label,
)
from evaluation.records import scored_runs, ScoredRun

_TOTALLER_ONLY = VariantConfig(totaller_enabled=True, reflection_enabled=False)
_REFLECTION_ONLY = VariantConfig(totaller_enabled=False, reflection_enabled=True)

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


_SOFT_METRIC = ("Soft", "aggregate")
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
        cells.append(f"{value:.3f}" if value is not None else "{--}")
    return cells


def _model_parts(model: str) -> tuple[str, str]:
    spec = LlmSpec.parse(model)
    short = spec.model.rsplit("/", 1)[-1]
    return short, _resolved_effort(model)


def _summary_table(
    records_by_judge: dict[str, list[ScoredRun]],
    *,
    key: str,
    first_header: str,
    order: Callable[[set[str]], list[str]],
    row_label: Callable[[str], str],
) -> str:
    """Both summary tables carry the same metrics in the same order; only the grouping differs."""
    groups = _group_by(primary_records(records_by_judge), key)
    judge_columns = _judge_columns(records_by_judge, key, (_SOFT_METRIC, _SAFETY_METRIC))

    numeric = [r"MAE~[\%]", r"MSE~[\unit{\percent\squared}]", "Iter.", "Calls", r"Wall-clock~[s]"]
    numeric += [head for head, _ in judge_columns]
    headers = [first_header] + [f"{{{head}}}" for head in numeric]
    spec = "l" + "S[table-format=2.2]" + "S[table-format=4.2]" + "S[table-format=1.2]" * 2 + "S[table-format=3.1]"
    spec += "S[table-format=1.3]" * len(judge_columns)
    lines = [
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for label in order(set(groups)):
        group = groups[label]
        cells = [
            row_label(label),
            f"{statistics.fmean(r.mae_pct for r in group):.2f}",
            f"{statistics.fmean(r.mse_pct for r in group):.2f}",
            f"{statistics.fmean(r.iterations for r in group):.2f}",
            f"{statistics.fmean(r.totaller_calls for r in group):.2f}",
            f"{statistics.fmean(r.elapsed_s for r in group):.1f}",
            *_judge_cells(judge_columns, label),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def variant_summary_table(records_by_judge: dict[str, list[ScoredRun]]) -> str:
    """One row per ablation variant, averaged over all runs."""
    return _summary_table(
        records_by_judge,
        key="variant",
        first_header="Variant",
        order=_variant_order,
        row_label=lambda label: rf"\texttt{{{_escape(variant_display_label(label))}}}",
    )


def model_summary_table(records_by_judge: dict[str, list[ScoredRun]]) -> str:
    """One row per model-and-effort configuration under the full variant."""

    def _order(labels: set[str]) -> list[str]:
        return [model for model in _model_order(labels) if model in labels]

    def _row_label(model: str) -> str:
        name, effort = _model_parts(model)
        return rf"\texttt{{{_escape(name)}}} & {_escape(effort)}"

    groups = _group_by(primary_records(records_by_judge), "llm_model")
    judge_columns = _judge_columns(records_by_judge, "llm_model", (_SOFT_METRIC, _SAFETY_METRIC))
    numeric = [r"MAE~[\%]", r"MSE~[\unit{\percent\squared}]", "Iter.", "Calls", r"Wall-clock~[s]"]
    numeric += [head for head, _ in judge_columns]
    headers = ["Model", "Effort"] + [f"{{{head}}}" for head in numeric]
    spec = "ll" + "S[table-format=2.2]" + "S[table-format=4.2]" + "S[table-format=1.2]" * 2 + "S[table-format=3.1]"
    spec += "S[table-format=1.3]" * len(judge_columns)
    lines = [
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for label in _order(set(groups)):
        group = groups[label]
        cells = [
            _row_label(label),
            f"{statistics.fmean(r.mae_pct for r in group):.2f}",
            f"{statistics.fmean(r.mse_pct for r in group):.2f}",
            f"{statistics.fmean(r.iterations for r in group):.2f}",
            f"{statistics.fmean(r.totaller_calls for r in group):.2f}",
            f"{statistics.fmean(r.elapsed_s for r in group):.1f}",
            *_judge_cells(judge_columns, label),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _format_p(test: SignTest) -> str:
    if test.p_value is None:
        return "{--}"
    return f"{test.p_value:.3f}"


def _sign_test_row(comparison: str, metric_label: str, test: SignTest) -> str:
    n_pairs = len(test.deltas)
    n_ties = n_pairs - test.n_effective
    cells = [
        comparison,
        metric_label,
        f"{test.n_improved}/{test.n_effective}",
        str(n_ties),
        _format_p(test),
    ]
    return " & ".join(cells) + r" \\"


def paired_sign_test_table(records_by_judge: dict[str, list[ScoredRun]]) -> str:
    """Exact one-sided sign tests on per-scenario cell means."""
    primary = primary_records(records_by_judge)
    baseline = variant_display_label(BASELINE.label)
    totaller = variant_display_label(_TOTALLER_ONLY.label)
    reflection = variant_display_label(_REFLECTION_ONLY.label)
    full = variant_display_label(FULL_VARIANT.label)
    comparisons: list[tuple[str, str, str]] = [
        (rf"\texttt{{{baseline}}} $\to$ \texttt{{{totaller}}}", BASELINE.label, _TOTALLER_ONLY.label),
        (rf"\texttt{{{baseline}}} $\to$ \texttt{{{reflection}}}", BASELINE.label, _REFLECTION_ONLY.label),
        (rf"\texttt{{{totaller}}} $\to$ \texttt{{{full}}}", _TOTALLER_ONLY.label, FULL_VARIANT.label),
        (rf"\texttt{{{reflection}}} $\to$ \texttt{{{full}}}", _REFLECTION_ONLY.label, FULL_VARIANT.label),
    ]
    metrics: list[tuple[str, Metric, list[ScoredRun]]] = [("MAE", MAE_METRIC, primary)]
    for judge in all_judges():
        judge_records = records_by_judge.get(judge.key, [])
        metrics.append((f"Soft {judge.label}", SOFT_METRIC, judge_records))
        metrics.append((f"Safety {judge.label}", SAFETY_METRIC, judge_records))

    lines = [
        r"\begin{tabular}{llrrS[table-format=1.3]}",
        r"\toprule",
        r"Comparison & Metric & {Improved} & {Ties} & {$p$} \\",
        r"\midrule",
    ]
    for index, (comparison, baseline_variant, treatment_variant) in enumerate(comparisons):
        if index:
            lines.append(r"\midrule")
        for metric_label, metric, records in metrics:
            test = paired_sign_test(
                records,
                metric,
                baseline_variant=baseline_variant,
                treatment_variant=treatment_variant,
            )
            lines.append(_sign_test_row(comparison, metric_label, test))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_scenario_mae_table(records: list[ScoredRun]) -> str:
    """Mean macro error for every (scenario, variant) pair of a single-model ablation grid."""
    variants = _variant_order({r.variant for r in records})
    scenarios = sorted({r.scenario_id for r in records})

    by_pair: dict[tuple[str, str], list[float]] = {}
    for record in records:
        by_pair.setdefault((record.scenario_id, record.variant), []).append(record.mae_pct)

    col_spec = "l" + r"S[table-format=2.1]" * len(variants)
    header = (
        "Scenario & " + " & ".join(rf"{{\texttt{{{_escape(variant_display_label(v))}}}}}" for v in variants) + r" \\"
    )
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
            cells.append(f"{statistics.fmean(values):.1f}" if values else "{--}")
        scenario_tex = _escape(scenario).replace("-", r"-\allowbreak ")
        lines.append(rf"\texttt{{{scenario_tex}}} & " + " & ".join(cells) + r" \\")
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
                ("paired_sign_tests.tex", paired_sign_test_table(ablation_scores)),
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
