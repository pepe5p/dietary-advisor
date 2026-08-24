"""`evaluation` Typer CLI: collect case runs, then score stored results."""

from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Collect dietary-advisor case runs and evaluate stored results.")

console = Console()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command("run-cases")
def run_cases(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Execute remaining (model, variant, scenario) runs into the configured output dir."""
    _configure_logging(verbose)
    from evaluation.case_runner import collect_runs, is_done, planned_runs

    specs = planned_runs()
    already_done = sum(1 for s in specs if is_done(s))
    remaining = len(specs) - already_done
    console.print(
        f"Planned [bold]{len(specs)}[/bold] runs "
        f"([green]{already_done}[/green] already done, "
        f"[cyan]{remaining}[/cyan] remaining)."
    )
    if remaining == 0:
        console.print("[green]Nothing to do.[/green]")
        return

    summary = asyncio.run(collect_runs(specs))
    console.print(
        f"[green]Succeeded[/green] {summary.succeeded} / "
        f"[red]failed[/red] {summary.failed} "
        f"(of {summary.attempted} attempted; {summary.already_done} were already done)."
    )
    if summary.failed:
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    force: bool = typer.Option(False, "--force", "-f", help="Rescore runs that already have a score file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score stored case-run records and print a per-variant summary."""
    _configure_logging(verbose)
    from evaluation.case_runner import is_done, planned_runs
    from evaluation.judges import all_judges, JUDGE_REPS
    from evaluation.plotting import primary_records
    from evaluation.records import scored_runs
    from evaluation.scoring import score_runs, scored_reps, summarize

    specs = planned_runs()
    stored = sum(1 for s in specs if is_done(s))
    missing = len(specs) - stored

    console.print(
        f"Planned [bold]{len(specs)}[/bold] runs ([green]{stored}[/green] stored, [yellow]{missing}[/yellow] missing)."
    )
    if missing:
        console.print("[yellow]Some planned runs have no stored record.[/yellow] Collect them with `run-cases` first.")

    any_to_score = False
    for judge in all_judges():
        done_specs = [s for s in specs if is_done(s)]
        already = sum(len(scored_reps(s, judge=judge)) for s in done_specs)
        total_verdicts = len(done_specs) * JUDGE_REPS
        to_score = total_verdicts - already if not force else total_verdicts
        console.print(
            f"[bold]{judge.key}[/bold] ({judge.model_id}): "
            f"[cyan]{already}[/cyan] / {total_verdicts} verdicts scored, "
            f"[cyan]{to_score}[/cyan] to score."
        )
        if to_score > 0 or force:
            any_to_score = True

    if any_to_score:
        summaries = asyncio.run(score_runs(specs, force=force))
        for judge in all_judges():
            summary = summaries[judge.key]
            console.print(
                f"[bold]{judge.key}[/bold]: "
                f"[green]Succeeded[/green] {summary.succeeded} / "
                f"[red]failed[/red] {summary.failed} "
                f"(of {summary.attempted} verdicts attempted; {summary.already_scored} were already scored)."
            )
        if any(s.failed for s in summaries.values()):
            raise typer.Exit(code=1)
    else:
        console.print("[green]Nothing to score.[/green]")

    records_by_judge = {judge.key: [record for _, record in scored_runs(specs, judge=judge)] for judge in all_judges()}
    primary = primary_records(records_by_judge)
    if not primary:
        console.print("[yellow]No score records available yet.[/yellow]")
        return

    summaries_by_judge = {
        judge.key: {(row.llm_model, row.variant): row for row in summarize(records_by_judge[judge.key])}
        for judge in all_judges()
    }

    table = Table(title="Variant summary")
    table.add_column("Model")
    table.add_column("Variant")
    table.add_column("No. Runs", justify="right")
    table.add_column("Avg. MAE %", justify="right")
    table.add_column("Avg. Elapsed s", justify="right")
    table.add_column("Avg. Iterations", justify="right")
    for judge in all_judges():
        table.add_column(f"Avg. Soft {judge.label}", justify="right")
        table.add_column(f"Avg. Safety {judge.label}", justify="right")
        table.add_column(f"No. Violations {judge.label}", justify="right")

    for row in summarize(primary):
        key = (row.llm_model, row.variant)
        judge_cells: list[str] = []
        for judge in all_judges():
            judge_row = summaries_by_judge[judge.key].get(key)
            judge_cells += (
                [
                    f"{judge_row.soft_aggregate:.4f}",
                    f"{judge_row.safety_adherence:.4f}",
                    str(judge_row.n_safety_violations),
                ]
                if judge_row
                else ["--", "--", "--"]
            )
        table.add_row(
            row.llm_model,
            row.variant,
            str(row.n_runs),
            f"{row.mae_pct:.2f}",
            f"{row.elapsed_s:.2f}",
            f"{row.iterations:.2f}",
            *judge_cells,
        )
    console.print(table)


def _plot_experiment(experiment: str) -> bool:
    from evaluation.case_runner.grid import experiment_runs
    from evaluation.judges import all_judges
    from evaluation.plotting import primary_records, render_experiment
    from evaluation.records import scored_runs
    from evaluation.reporting import write_experiment_metrics

    specs = experiment_runs(experiment)
    records_by_judge = {judge.key: [record for _, record in scored_runs(specs, judge=judge)] for judge in all_judges()}

    scored_per_judge = ", ".join(f"{judge.key} {len(records_by_judge[judge.key])}" for judge in all_judges())
    console.print(f"Experiment [bold]{experiment}[/bold]: {len(specs)} runs planned (scored: {scored_per_judge}).")
    if any(len(records_by_judge[judge.key]) < len(specs) for judge in all_judges()):
        console.print("[yellow]Some runs are not scored yet.[/yellow] Run `evaluate` first.")
    if not primary_records(records_by_judge):
        console.print("[red]No score records available.[/red]")
        return False

    paths = render_experiment(experiment, records_by_judge)
    for path in paths:
        console.print(f"Wrote {path}")
    metrics_path = write_experiment_metrics(experiment, records_by_judge)
    console.print(f"Wrote {metrics_path}")
    return True


def _plot_run_variability() -> bool:
    from evaluation.case_runner import planned_runs
    from evaluation.judges import all_judges
    from evaluation.plotting import (
        complete_spec_groups,
        model_variability,
        pooled_variability,
        render_run_variability,
        SOFT_METRIC,
        SPREAD_COLUMN_LABEL,
    )
    from evaluation.records import scored_runs

    specs = planned_runs()
    groups_by_judge = {judge.key: complete_spec_groups(scored_runs(specs, judge=judge)) for judge in all_judges()}

    scored = [(judge, groups_by_judge[judge.key]) for judge in all_judges() if groups_by_judge[judge.key]]
    if not scored:
        console.print("[yellow]No run specs with all three repetitions scored.[/yellow]")
        return False

    soft_metric = SOFT_METRIC
    soft_by_judge = {judge.key: model_variability(groups, soft_metric) for judge, groups in scored}
    soft_by_label = {judge.key: {entry.label: entry for entry in soft_by_judge[judge.key]} for judge, _ in scored}

    completeness = ", ".join(f"{judge.label} {len(groups)}" for judge, groups in scored)
    console.print(f"Run variability: complete specs per judge: [bold]{completeness}[/bold].")

    table = Table(title=f"{soft_metric.title} run spread by model -- {SPREAD_COLUMN_LABEL}")
    table.add_column("Model", no_wrap=True)
    for judge, _ in scored:
        table.add_column(f"Specs ({judge.label})", justify="right")
        table.add_column(f"Spread ({judge.label})", justify="right")

    seen: list[str] = []
    for judge, _ in scored:
        seen += [label for label in soft_by_label[judge.key] if label not in seen]

    for label in seen:
        cells: list[str] = []
        for judge, _ in scored:
            entry = soft_by_label[judge.key].get(label)
            cells += [str(entry.n_specs), f"{entry.mean:.3f}"] if entry else ["--", "--"]
        table.add_row(label, *cells)

    pooled_cells: list[str] = []
    for _, groups in scored:
        pooled = pooled_variability(groups, soft_metric)
        pooled_cells += [str(pooled.n_specs), f"{pooled.mean:.3f}"]
    table.add_row("all", *pooled_cells)
    console.print(table)

    paths = render_run_variability(groups_by_judge)
    for path in paths:
        console.print(f"Wrote {path}")
    return True


@app.command()
def plot(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Render per-metric bar charts and metrics JSON for all experiments' scored runs."""
    _configure_logging(verbose)
    from evaluation.case_runner.grid import EXPERIMENTS

    wrote_any = False
    for experiment in sorted(EXPERIMENTS):
        if _plot_experiment(experiment):
            wrote_any = True
    if _plot_run_variability():
        wrote_any = True

    if not wrote_any:
        raise typer.Exit(code=1)


@app.command()
def tables(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Render booktabs LaTeX tables for the thesis from currently scored runs."""
    _configure_logging(verbose)
    from evaluation.latex import write_tables

    paths = write_tables()
    if not paths:
        console.print("[yellow]No score records available yet.[/yellow] Run `evaluate` first.")
        raise typer.Exit(code=1)
    for path in paths:
        console.print(f"Wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
