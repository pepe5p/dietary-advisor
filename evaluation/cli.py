"""`evaluation` Typer CLI: collect case runs, then (later) score them."""

from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console

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
    """Execute remaining (model, variant, scenario) runs into `outputs/`."""
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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score existing `outputs/` records (pending the scoring-stage rework)."""
    _configure_logging(verbose)
    console.print(
        "[yellow]Scoring stage is pending rework.[/yellow] "
        "Collect runs with `run-cases` first; evaluation against stored "
        "results will land in a later change."
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
