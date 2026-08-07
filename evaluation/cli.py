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
    from evaluation.scoring import is_scored, load, score_runs, summarize

    specs = planned_runs()
    stored = sum(1 for s in specs if is_done(s))
    missing = len(specs) - stored
    already_scored = sum(1 for s in specs if is_done(s) and is_scored(s))
    to_score = stored - already_scored if not force else stored

    console.print(
        f"Planned [bold]{len(specs)}[/bold] runs "
        f"([green]{stored}[/green] stored, "
        f"[yellow]{missing}[/yellow] missing, "
        f"[cyan]{already_scored}[/cyan] already scored)."
    )
    if missing:
        console.print("[yellow]Some planned runs have no stored record.[/yellow] Collect them with `run-cases` first.")
    if to_score == 0 and not force:
        console.print("[green]Nothing to score.[/green]")
    else:
        summary = asyncio.run(score_runs(specs, force=force))
        console.print(
            f"[green]Succeeded[/green] {summary.succeeded} / "
            f"[red]failed[/red] {summary.failed} "
            f"(of {summary.attempted} attempted; {summary.already_scored} were already scored)."
        )
        if summary.failed:
            raise typer.Exit(code=1)

    scored_records = [load(s) for s in specs if is_scored(s)]
    if not scored_records:
        console.print("[yellow]No score records available yet.[/yellow]")
        return

    table = Table(title="Variant summary")
    table.add_column("Model")
    table.add_column("Variant")
    table.add_column("No. Runs", justify="right")
    table.add_column("Avg. MAE %", justify="right")
    table.add_column("Avg. Soft", justify="right")
    table.add_column("Avg. Safety", justify="right")
    table.add_column("Avg. Iterations", justify="right")
    table.add_column("Avg. Elapsed s", justify="right")
    table.add_column("No. Violations", justify="right")

    for row in summarize(scored_records):
        table.add_row(
            row.llm_model,
            row.variant,
            str(row.n_runs),
            f"{row.mae_pct:.2f}",
            f"{row.soft_aggregate:.4f}",
            f"{row.safety_adherence:.4f}",
            f"{row.iterations:.2f}",
            f"{row.elapsed_s:.2f}",
            str(row.n_safety_violations),
        )
    console.print(table)


def _plot_experiment(experiment: str) -> bool:
    from evaluation.case_runner.grid import experiment_runs
    from evaluation.plotting import render_experiment
    from evaluation.reporting import write_experiment_metrics
    from evaluation.scoring import is_scored, load

    specs = experiment_runs(experiment)
    scored_specs = [spec for spec in specs if is_scored(spec)]
    missing = len(specs) - len(scored_specs)

    console.print(
        f"Experiment [bold]{experiment}[/bold]: [green]{len(scored_specs)}[/green] / {len(specs)} runs scored."
    )
    if missing:
        console.print("[yellow]Some runs are not scored yet.[/yellow] Run `evaluate` first.")
    if not scored_specs:
        console.print("[red]No score records available.[/red]")
        return False

    records = [load(spec) for spec in scored_specs]
    paths = render_experiment(experiment, records)
    for path in paths:
        console.print(f"Wrote {path}")
    metrics_path = write_experiment_metrics(experiment, records)
    console.print(f"Wrote {metrics_path}")
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

    if not wrote_any:
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
