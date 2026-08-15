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
    from evaluation.judges import all_judges
    from evaluation.plotting import primary_records
    from evaluation.scoring import is_scored, load, score_runs, summarize

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
        already_scored = sum(1 for s in specs if is_done(s) and is_scored(s, judge=judge))
        to_score = stored - already_scored if not force else stored
        console.print(
            f"[bold]{judge.key}[/bold] ({judge.model_id}): "
            f"[cyan]{already_scored}[/cyan] already scored, "
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
                f"(of {summary.attempted} attempted; {summary.already_scored} were already scored)."
            )
        if any(s.failed for s in summaries.values()):
            raise typer.Exit(code=1)
    else:
        console.print("[green]Nothing to score.[/green]")

    records_by_judge = {
        judge.key: [load(s, judge=judge) for s in specs if is_scored(s, judge=judge)] for judge in all_judges()
    }
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
            summary = summaries_by_judge[judge.key].get(key)
            judge_cells += (
                [
                    f"{summary.soft_aggregate:.4f}",
                    f"{summary.safety_adherence:.4f}",
                    str(summary.n_safety_violations),
                ]
                if summary
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
    from evaluation.reporting import write_experiment_metrics
    from evaluation.scoring import is_scored, load

    specs = experiment_runs(experiment)
    records_by_judge = {
        judge.key: [load(spec, judge=judge) for spec in specs if is_scored(spec, judge=judge)]
        for judge in all_judges()
    }

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
        METRICS,
        render_run_variability,
        scenario_spread_stats,
        sem_for_reps,
    )
    from evaluation.scoring import is_scored, load

    specs = planned_runs()
    groups_by_judge = {
        judge.key: complete_spec_groups(
            [(spec, load(spec, judge=judge)) for spec in specs if is_scored(spec, judge=judge)]
        )
        for judge in all_judges()
    }

    scored = [(judge, groups_by_judge[judge.key]) for judge in all_judges() if groups_by_judge[judge.key]]
    if not scored:
        console.print("[yellow]No run specs with all three repetitions scored.[/yellow]")
        return False

    primary_judge, primary_groups = scored[0]
    mae_stats = scenario_spread_stats(primary_groups, METRICS[0])
    soft_stats_by_judge = {judge.key: scenario_spread_stats(groups, METRICS[1]) for judge, groups in scored}

    console.print(
        f"Run variability: [bold]{len(primary_groups)}[/bold] complete specs ({primary_judge.label})."
    )
    table = Table(title="Run variability by scenario")
    table.add_column("Scenario")
    table.add_column("Specs", justify="right")
    table.add_column("MAE mean", justify="right")
    table.add_column("MAE pooled std", justify="right")
    table.add_column("MAE SEM@3", justify="right")
    for judge, _ in scored:
        table.add_column(f"Soft {judge.label} mean", justify="right")
        table.add_column(f"Soft {judge.label} pooled std", justify="right")
        table.add_column(f"Soft {judge.label} SEM@3", justify="right")

    soft_by_label = {
        judge.key: {entry.label: entry for entry in soft_stats_by_judge[judge.key]} for judge, _ in scored
    }
    for mae in mae_stats:
        soft_cells: list[str] = []
        for judge, _ in scored:
            soft = soft_by_label[judge.key].get(mae.label)
            soft_cells += (
                [
                    f"{soft.grand_mean:.3f}",
                    f"{soft.pooled_std:.3f}",
                    f"{sem_for_reps(soft.pooled_std, 3):.3f}",
                ]
                if soft
                else ["--", "--", "--"]
            )
        table.add_row(
            mae.label,
            str(mae.n_specs),
            f"{mae.grand_mean:.2f}",
            f"{mae.pooled_std:.2f}",
            f"{sem_for_reps(mae.pooled_std, 3):.2f}",
            *soft_cells,
        )
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
