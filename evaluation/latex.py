"""Render scored evaluation runs as booktabs LaTeX tables for the thesis.

Each function returns bare ``tabular`` content (no ``table`` float, caption or
label) so the thesis chapters control placement, captions and labels; only the
numbers are generated here.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from evaluation.case_runner.grid import experiment_runs, EXPERIMENTS
from evaluation.case_runner.store import load as load_run
from evaluation.case_runner.store import RunRecord
from evaluation.judges import all_judges
from evaluation.plotting.stats import (
    _bare_short_model_name,
    _short_model_name,
    _variant_order,
    primary_records,
)
from evaluation.scoring.store import is_scored, ScoreRecord
from evaluation.scoring.store import load as load_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TABLES_DIR = _REPO_ROOT / "thesis" / "tables"

# USD per 1M tokens (input, output, cached input), published provider pricing at evaluation time.
_MODEL_PRICING_USD_PER_1M: dict[str, tuple[float, float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.6-flash": (0.75, 3.75, 0.075),
    "gemini-3.7-flash": (0.375, 1.875, 0.0375),
    "gpt-5.6-luna": (0.10, 0.60, 0.01),
}


def _escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _scored_records_by_judge(experiment: str) -> dict[str, list[ScoreRecord]]:
    specs = experiment_runs(experiment)
    return {
        judge.key: [load_score(spec, judge=judge) for spec in specs if is_scored(spec, judge=judge)]
        for judge in all_judges()
    }


def _run_records(experiment: str) -> list[RunRecord]:
    specs = experiment_runs(experiment)
    return [
        load_run(spec) for spec in specs if any(is_scored(spec, judge=judge) for judge in all_judges())
    ]


def _group_by(records: list[ScoreRecord], key: str) -> dict[str, list[ScoreRecord]]:
    groups: dict[str, list[ScoreRecord]] = {}
    for record in records:
        groups.setdefault(getattr(record, key), []).append(record)
    return groups


def _soft_by_key(records: list[ScoreRecord], key: str) -> dict[str, float]:
    groups = _group_by(records, key)
    return {label: statistics.fmean(r.qualitative.aggregate for r in group) for label, group in groups.items()}


def _soft_columns(records_by_judge: dict[str, list[ScoreRecord]], key: str) -> list[tuple[str, dict[str, float]]]:
    """One (header, per-group mean) column per registered judge, in registry order."""
    return [
        (rf"Soft agg.\ {_escape(judge.label)}", _soft_by_key(records_by_judge.get(judge.key, []), key))
        for judge in all_judges()
    ]


def _soft_cells(columns: list[tuple[str, dict[str, float]]], group_key: str) -> list[str]:
    cells = []
    for _, means in columns:
        value = means.get(group_key)
        cells.append(f"{value:.3f}" if value is not None else "--")
    return cells


def variant_summary_table(records_by_judge: dict[str, list[ScoreRecord]]) -> str:
    """One row per ablation variant: error, preference and cost metrics, averaged over all runs."""
    groups = _group_by(primary_records(records_by_judge), "variant")
    order = _variant_order(set(groups))
    soft_columns = _soft_columns(records_by_judge, "variant")

    headers = ["Variant", "$N$", r"MAE~[\%]", r"MSE~[\%]", *(head for head, _ in soft_columns)]
    headers += ["Safety", "Iter.", r"Elapsed~[s]"]
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
            str(len(group)),
            f"{statistics.fmean(r.mae_pct for r in group):.2f}",
            f"{statistics.fmean(r.mse_pct for r in group):.2f}",
            *_soft_cells(soft_columns, label),
            f"{statistics.fmean(r.qualitative.safety_adherence for r in group):.3f}",
            f"{statistics.fmean(r.iterations for r in group):.2f}",
            f"{statistics.fmean(r.elapsed_s for r in group):.1f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def model_summary_table(records_by_judge: dict[str, list[ScoreRecord]]) -> str:
    """One row per model under the full variant: error, preference and cost metrics."""
    groups = _group_by(primary_records(records_by_judge), "llm_model")
    soft_columns = _soft_columns(records_by_judge, "llm_model")

    headers = ["Model", "$N$", r"MAE~[\%]", *(head for head, _ in soft_columns), "Iter.", r"Elapsed~[s]"]
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
            str(len(group)),
            f"{statistics.fmean(r.mae_pct for r in group):.2f}",
            *_soft_cells(soft_columns, model),
            f"{statistics.fmean(r.iterations for r in group):.2f}",
            f"{statistics.fmean(r.elapsed_s for r in group):.1f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_scenario_mae_table(records: list[ScoreRecord]) -> str:
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


def token_latency_table(run_records: list[RunRecord]) -> str:
    """Mean token usage, request count and wall time per ablation variant."""
    groups: dict[str, list[RunRecord]] = {}
    for record in run_records:
        groups.setdefault(record.variant, []).append(record)
    order = _variant_order(set(groups))

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & Input tok. & Output tok. & Requests & Elapsed~[s] \\",
        r"\midrule",
    ]
    for label in order:
        group = groups[label]
        input_tok = statistics.fmean(r.telemetry.get("input_tokens", 0) for r in group)
        output_tok = statistics.fmean(r.telemetry.get("output_tokens", 0) for r in group)
        requests = statistics.fmean(r.telemetry.get("requests", 0) for r in group)
        elapsed = statistics.fmean(r.elapsed_s for r in group)
        lines.append(
            f"{_escape(label)} & {input_tok:,.0f} & {output_tok:,.0f} & {requests:.1f} & {elapsed:.1f} \\\\".replace(
                ",", r"\,"
            ),
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def model_token_cost_table(run_records: list[RunRecord]) -> str:
    """Mean token usage and estimated USD cost per run, one row per model, under the full variant."""
    groups: dict[str, list[RunRecord]] = {}
    for record in run_records:
        groups.setdefault(_short_model_name(record.llm_model), []).append(record)

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Input tok. & Output tok. & Requests & Cost per run~[\$] \\",
        r"\midrule",
    ]
    for model in sorted(groups):
        group = groups[model]
        input_tok = statistics.fmean(r.telemetry.get("input_tokens", 0) for r in group)
        cache_tok = statistics.fmean(r.telemetry.get("cache_read_tokens", 0) for r in group)
        output_tok = statistics.fmean(r.telemetry.get("output_tokens", 0) for r in group)
        requests = statistics.fmean(r.telemetry.get("requests", 0) for r in group)
        price_in, price_out, price_cached = _MODEL_PRICING_USD_PER_1M.get(
            _bare_short_model_name(group[0].llm_model),
            (0.0, 0.0, 0.0),
        )
        uncached_tok = max(input_tok - cache_tok, 0.0)
        cost = uncached_tok / 1e6 * price_in + cache_tok / 1e6 * price_cached + output_tok / 1e6 * price_out
        lines.append(
            f"{_escape(model)} & {input_tok:,.0f} & {output_tok:,.0f} & {requests:.1f} & {cost:.4f} \\\\".replace(
                ",", r"\,"
            ),
        )
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

            ablation_runs = _run_records("ablation")
            if ablation_runs:
                path = dest / "ablation_token_latency.tex"
                path.write_text(token_latency_table(ablation_runs) + "\n", encoding="utf-8")
                written.append(path)

    if "models" in EXPERIMENTS:
        model_scores = _scored_records_by_judge("models")
        if primary_records(model_scores):
            path = dest / "model_comparison_summary.tex"
            path.write_text(model_summary_table(model_scores) + "\n", encoding="utf-8")
            written.append(path)

            model_runs = _run_records("models")
            if model_runs:
                path = dest / "model_comparison_token_cost.tex"
                path.write_text(model_token_cost_table(model_runs) + "\n", encoding="utf-8")
                written.append(path)

    return written
