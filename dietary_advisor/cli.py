"""`dietary-advisor` Typer CLI: recommend / evaluate / ingest-corpus / profile."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dietary_advisor.config import get_settings
from dietary_advisor.knowledge.ingest import ingest_corpus
from dietary_advisor.pipeline import Pipeline, VARIANTS
from dietary_advisor.profile_manager.service import ProfileService
from dietary_advisor.profile_manager.store import ProfileStore
from dietary_advisor.schemas.profile import UserProfile

app = typer.Typer(help="Neuro-symbolic dietary advisor (master's thesis CLI).")
profile_app = typer.Typer(help="Manage user profiles in the local SQLite store.")
app.add_typer(profile_app, name="profile")

console = Console()
log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Top-level options."""
    _configure_logging(verbose)


# ---------- recommend ----------------------------------------------------------------------------


@app.command()
def recommend(
    profile_id: str = typer.Argument(..., help="user_id of a profile already in the store."),
    query: str = typer.Option(
        "Plan one balanced day of meals.",
        "--query",
        "-q",
        help="The user-facing prompt forwarded to the LLM.",
    ),
    variant: str = typer.Option("V4", "--variant", help=f"One of {sorted(VARIANTS)}"),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the full result to JSON."),
) -> None:
    """Generate a single recommendation for an existing profile."""
    if variant not in VARIANTS:
        raise typer.BadParameter(f"Unknown variant {variant!r}. Available: {sorted(VARIANTS)}")
    service = ProfileService.default()
    profile = service.get(profile_id)
    if profile is None:
        raise typer.BadParameter(f"No profile with id {profile_id!r}. Use `profile list` to inspect.")

    with Pipeline(variant, profile_service=service) as pipeline:
        result = asyncio.run(pipeline.run(profile, query))

    _render_result(result)
    if json_out is not None:
        payload = {
            "variant": result.variant,
            "iterations": result.iterations,
            "plan": result.plan.model_dump(mode="json"),
            "report": result.report.model_dump(mode="json"),
            "targets": result.targets.model_dump(mode="json"),
            "constraints": [c.model_dump(mode="json") for c in result.constraints],
            "citations": [c.model_dump(mode="json") for c in result.citations],
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {json_out}")


def _render_result(result: object) -> None:
    """Pretty-print a `PipelineResult` to the console."""
    from dietary_advisor.pipeline import PipelineResult  # local import to avoid cycles

    assert isinstance(result, PipelineResult)
    console.rule(f"Variant {result.variant} - reflection iterations: {result.iterations}")

    targets_tbl = Table(title="Macro targets", show_header=True)
    targets_tbl.add_column("kcal")
    targets_tbl.add_column("protein g")
    targets_tbl.add_column("carbs g")
    targets_tbl.add_column("fat g")
    targets_tbl.add_column("fiber g")
    targets_tbl.add_row(
        f"{result.targets.energy_kcal:.0f}",
        f"{result.targets.protein_g:.0f}",
        f"{result.targets.carbs_g:.0f}",
        f"{result.targets.fat_g:.0f}",
        f"{result.targets.fiber_g:.0f}",
    )
    console.print(targets_tbl)

    plan_tbl = Table(title=f"MealPlan ({len(result.plan.meals)} meals)")
    plan_tbl.add_column("Meal")
    plan_tbl.add_column("Recipe")
    plan_tbl.add_column("Portions (g)")
    for meal in result.plan.meals:
        portions = ", ".join(f"{p.food.name}: {p.grams:.0f}" for p in meal.recipe.portions)
        plan_tbl.add_row(meal.kind.value, meal.recipe.name, portions)
    console.print(plan_tbl)

    if result.plan.rationale:
        console.print(f"\n[bold]Rationale[/bold]: {result.plan.rationale}")

    status = "[green]OK[/green]" if result.report.hard_satisfied else "[red]FAIL[/red]"
    console.print(f"\nValidation: {status} (HSR={result.report.hsr})")
    if result.report.violations:
        for v in result.report.violations:
            console.print(f"  [red]-[/red] {v.detail}")
    if result.citations:
        console.print(f"\n[bold]Citations[/bold]: {len(result.citations)} chunks retrieved.")


# ---------- evaluate ----------------------------------------------------------------------------


@app.command()
def evaluate(
    variants: str = typer.Option("V0,V1,V2,V3,V4", "--variants", help="Comma-separated variant ids."),
    levels: str = typer.Option("1,2,3", "--levels", help="Comma-separated complexity levels."),
    output: Path = typer.Option(Path("evaluation/reports/ablation.csv"), "--output"),
    repeats: int = typer.Option(1, "--repeats", help="Repeat each (variant,profile,query) N times."),
    profile_dir: Path = typer.Option(Path("evaluation/profiles"), "--profile-dir"),
) -> None:
    """Run the full ablation grid and write a CSV (and Markdown summary) report."""
    # Local imports keep `cli --help` fast even when matplotlib isn't built.
    from evaluation.ablation import variants_from_ids
    from evaluation.report import write_markdown_summary, write_plots
    from evaluation.runner import run_ablation_grid

    variant_ids = [v.strip() for v in variants.split(",") if v.strip()]
    level_ids = [int(s) for s in levels.split(",") if s.strip()]
    chosen = variants_from_ids(variant_ids)

    df = asyncio.run(
        run_ablation_grid(
            variants=chosen,
            levels=level_ids,
            repeats=repeats,
            profile_dir=profile_dir,
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


# ---------- ingest-corpus ----------------------------------------------------------------------


@app.command("ingest-corpus")
def ingest_corpus_cmd(
    force: bool = typer.Option(False, "--force", help="Re-download every PDF even if cached."),
) -> None:
    """Download (best-effort) and index the clinical-guideline corpus."""
    stats = ingest_corpus(force_redownload=force)
    console.rule("Corpus ingest stats")
    console.print(f"Downloaded: {stats.fetched or '-'}")
    console.print(f"Seed-only:  {stats.seeded or '-'}")
    console.print(f"Failed:     {stats.failed or '-'}")
    console.print(f"Chunks indexed: [bold]{stats.chunks_indexed}[/bold]")


# ---------- profile -----------------------------------------------------------------------------


@profile_app.command("list")
def profile_list() -> None:
    """List all profiles in the local store."""
    store = ProfileStore()
    profiles = store.list_all()
    if not profiles:
        console.print("[yellow]No profiles in store. Use `profile add` to create one.[/yellow]")
        return
    tbl = Table(title="Profiles")
    tbl.add_column("user_id")
    tbl.add_column("name")
    tbl.add_column("level")
    tbl.add_column("conditions")
    tbl.add_column("allergens")
    for p in profiles:
        tbl.add_row(
            p.user_id,
            p.name or "-",
            str(p.complexity_level),
            ", ".join(c.value for c in p.conditions) or "-",
            ", ".join(a.value for a in p.allergens) or "-",
        )
    console.print(tbl)


@profile_app.command("add")
def profile_add(
    json_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Import a single profile JSON file into the store."""
    store = ProfileStore()
    profile: UserProfile = store.import_json(json_path.read_text(encoding="utf-8"))
    console.print(f"[green]Stored[/green] profile {profile.user_id} (level {profile.complexity_level})")


@profile_app.command("import-dir")
def profile_import_dir(
    dir_path: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
) -> None:
    """Bulk-import every *.json file in a directory."""
    store = ProfileStore()
    profiles = store.import_dir(dir_path)
    console.print(f"[green]Imported[/green] {len(profiles)} profiles from {dir_path}")


@profile_app.command("show")
def profile_show(user_id: str) -> None:
    """Print a profile as JSON."""
    store = ProfileStore()
    payload = store.export_json(user_id)
    if payload is None:
        raise typer.BadParameter(f"No profile with id {user_id!r}.")
    console.print(payload)


@profile_app.command("delete")
def profile_delete(user_id: str) -> None:
    """Remove a profile from the store."""
    store = ProfileStore()
    if store.delete(user_id):
        console.print(f"[green]Deleted[/green] {user_id}")
    else:
        console.print(f"[yellow]No such profile[/yellow] {user_id}")


# ---------- info -------------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Print configuration sanity-check info."""
    settings = get_settings()
    tbl = Table(title="dietary-advisor settings")
    tbl.add_column("key")
    tbl.add_column("value")
    for k, v in settings.model_dump().items():
        tbl.add_row(k, str(v))
    console.print(tbl)


if __name__ == "__main__":  # pragma: no cover
    app()
