"""`evaluation` Typer CLI: run the leave-one-out ablation grid."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Run the neuro-symbolic dietary advisor ablation evaluation.")

console = Console()
log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def evaluate(
    output: Path = typer.Option(Path("evaluation/reports/ablation.csv"), "--output"),
    repeats: int = typer.Option(1, "--repeats", help="Repeat each (variant,profile,query) N times."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip G-Eval soft-preference scoring."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the leave-one-out ablation grid (full system + one config per disabled module)."""
    _configure_logging(verbose)
    # Local imports keep `evaluate --help` fast even when matplotlib isn't built.
    from evaluation.ablation import leave_one_out_variants
    from evaluation.report import write_markdown_summary, write_plots
    from evaluation.runner import run_ablation_grid

    df = asyncio.run(
        run_ablation_grid(
            variants=leave_one_out_variants(),
            repeats=repeats,
            run_judge=not no_judge,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    console.print(f"[green]Saved CSV[/green] {output} ({len(df)} rows)")

    summary_md = output.with_suffix(".md")
    write_markdown_summary(df, summary_md)
    console.print(f"[green]Saved summary[/green] {summary_md}")

    plot_path = output.with_suffix(".png")
    write_plots(df, plot_path)
    console.print(f"[green]Saved plots[/green] {plot_path}")
