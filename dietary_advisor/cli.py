"""`dietary-advisor` Typer CLI: recommend / evaluate."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from dietary_advisor.config import get_settings
from dietary_advisor.pipeline import Pipeline, VariantConfig
from dietary_advisor.profiles import get_profile
from dietary_advisor.schemas.nutrition import NutrientName
from dietary_advisor.schemas.profile import UserProfile
from dietary_advisor.tools.totaller import total_meal_plan

app = typer.Typer(help="Neuro-symbolic dietary advisor (master's thesis CLI).")

console = Console()
log = logging.getLogger(__name__)

# Set by the `--verbose`/`-v` top-level flag; gates the debug telemetry table
# (tool-call counts, LLM requests, token usage) printed by `_render_result`.
_verbose = False


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Top-level options."""
    global _verbose
    _verbose = verbose
    _configure_logging(verbose)


# ---------- recommend ----------------------------------------------------------------------------


def _parse_ingredients(raw: str | None) -> list[str]:
    """Split a comma-separated ingredient string into a clean list."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


_PROFILE_OPTION_HELP = (
    'Full UserProfile as a JSON string, e.g. \'{"user_id": "u1", "age": 30, "sex": "female", '
    '"height_cm": 165, "weight_kg": 60, "targets": {"energy_kcal": 2000, "protein_g": 90, '
    '"carbs_g": 250, "fat_g": 60}}\'. Mutually exclusive with --profile-id.'
)
_PROFILE_ID_OPTION_HELP = (
    "user_id of a built-in profile (see dietary_advisor/profiles.py). Mutually exclusive with --profile."
)


def _resolve_profile(profile: str | None, profile_id: str | None) -> UserProfile:
    """Resolve a `UserProfile` from the mutually exclusive --profile/--profile-id CLI inputs."""
    if (profile is None) == (profile_id is None):
        raise typer.BadParameter("Pass exactly one of --profile or --profile-id.")
    if profile_id is not None:
        try:
            return get_profile(profile_id)
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from None
    assert profile is not None
    try:
        return UserProfile.model_validate_json(profile)
    except ValidationError as exc:
        raise typer.BadParameter(f"Invalid --profile JSON: {exc}") from None


def _result_payload(result: object) -> dict[str, object]:
    """Serialise a `PipelineResult` to a JSON-friendly dict."""
    from dietary_advisor.pipeline import PipelineResult  # local import to avoid cycles

    assert isinstance(result, PipelineResult)
    return {
        "variant": result.variant,
        "iterations": result.iterations,
        "plan": result.plan.model_dump(mode="json"),
        "targets": result.targets.model_dump(mode="json"),
        "citations": [c.model_dump(mode="json") for c in result.citations],
        "shopping_list": result.shopping_list.model_dump(mode="json"),
        "telemetry": result.telemetry.as_dict(),
    }


@app.command()
def recommend(
    profile: str | None = typer.Option(None, "--profile", help=_PROFILE_OPTION_HELP),
    profile_id: str | None = typer.Option(None, "--profile-id", help=_PROFILE_ID_OPTION_HELP),
    query: str = typer.Option(
        "Plan one balanced day of meals.",
        "--query",
        "-q",
        help="The user-facing prompt forwarded to the LLM.",
    ),
    no_off: bool = typer.Option(False, "--no-off", help="Disable the Open Food Facts food database."),
    no_totaller: bool = typer.Option(False, "--no-totaller", help="Disable the deterministic totaller tool."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable clinical-guideline retrieval (RAG)."),
    no_reflective_loop: bool = typer.Option(
        False,
        "--no-reflective-loop",
        help="Disable the Generate-Review-Refine self-review loop.",
    ),
    available: str | None = typer.Option(
        None,
        "--available",
        "-a",
        help="Comma-separated ingredients to build the plan around (e.g. fridge contents).",
    ),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the full result to JSON."),
) -> None:
    """Generate a single recommendation for a profile (full system by default)."""
    resolved_profile = _resolve_profile(profile, profile_id)

    variant = VariantConfig(
        food_enabled=not no_off,
        totaller_enabled=not no_totaller,
        rag_enabled=not no_rag,
        reflection_enabled=not no_reflective_loop,
    )
    ingredients = _parse_ingredients(available)
    with Pipeline(variant) as pipeline:
        result = asyncio.run(pipeline.run(resolved_profile, query, available_ingredients=ingredients))

    _render_result(result)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(_result_payload(result), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {json_out}")


@app.command()
def chat(
    profile: str | None = typer.Option(None, "--profile", help=_PROFILE_OPTION_HELP),
    profile_id: str | None = typer.Option(None, "--profile-id", help=_PROFILE_ID_OPTION_HELP),
    query: str = typer.Option(
        "Plan one balanced day of meals.",
        "--query",
        "-q",
        help="The opening request that seeds the first plan.",
    ),
    no_off: bool = typer.Option(False, "--no-off", help="Disable the Open Food Facts food database."),
    no_totaller: bool = typer.Option(False, "--no-totaller", help="Disable the deterministic totaller tool."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable clinical-guideline retrieval (RAG)."),
    no_reflective_loop: bool = typer.Option(
        False,
        "--no-reflective-loop",
        help="Disable the Generate-Review-Refine self-review loop.",
    ),
    available: str | None = typer.Option(
        None,
        "--available",
        "-a",
        help="Comma-separated ingredients to build the plan around (e.g. fridge contents).",
    ),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the final result to JSON on exit."),
) -> None:
    """Interactively design and revise a plan over multiple turns (full system by default).

    The first turn generates a plan; every subsequent line is treated as a
    follow-up that revises the current plan (e.g. "make breakfast lower-carb").
    Enter 'quit' (or EOF) to stop.
    """
    resolved_profile = _resolve_profile(profile, profile_id)

    variant = VariantConfig(
        food_enabled=not no_off,
        totaller_enabled=not no_totaller,
        rag_enabled=not no_rag,
        reflection_enabled=not no_reflective_loop,
    )
    ingredients = _parse_ingredients(available)
    with Pipeline(variant) as pipeline:
        result = asyncio.run(pipeline.run(resolved_profile, query, available_ingredients=ingredients))
        _render_result(result)

        console.print("\n[dim]Type a follow-up to revise the plan, or 'quit' to finish.[/dim]")
        while True:
            try:
                line = console.input("\n[bold cyan]You[/bold cyan] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in {"quit", "exit", ":q"}:
                break
            result = asyncio.run(
                pipeline.run(
                    resolved_profile,
                    line,
                    available_ingredients=ingredients,
                    message_history=result.messages,
                ),
            )
            _render_result(result)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(_result_payload(result), indent=2), encoding="utf-8")
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

    plan_tbl = Table(title=f"MealPlan ({len(result.plan.meals)} meals)", show_lines=True)
    plan_tbl.add_column("Meal")
    plan_tbl.add_column("Meal Name")
    plan_tbl.add_column("Products")
    plan_tbl.add_column("Recipe")
    for meal in result.plan.meals:
        portions = ", ".join(f"{p.food.name}: {p.grams:.0f}" for p in meal.recipe.portions)
        plan_tbl.add_row(meal.kind.value, meal.recipe.name, portions, meal.recipe.instructions)
    console.print(plan_tbl)

    if result.plan.rationale:
        console.print(f"\n[bold]Rationale[/bold]: {result.plan.rationale}")

    if result.shopping_list.items:
        shop_tbl = Table(title="Shopping list")
        shop_tbl.add_column("Ingredient")
        shop_tbl.add_column("Source")
        shop_tbl.add_column("Total (g)", justify="right")
        shop_tbl.add_column("kcal", justify="right")
        shop_tbl.add_column("Protein g", justify="right")
        shop_tbl.add_column("Carbs g", justify="right")
        shop_tbl.add_column("Fat g", justify="right")
        for item in result.shopping_list.items:
            source = "[green]OFF DB[/green]" if item.from_open_food_facts else "[yellow]LLM est.[/yellow]"
            shop_tbl.add_row(
                item.name,
                source,
                f"{item.total_grams:.0f}",
                f"{item.energy_kcal:.0f}",
                f"{item.protein_g:.1f}",
                f"{item.carbs_g:.1f}",
                f"{item.fat_g:.1f}",
            )
        console.print(shop_tbl)

    totals = total_meal_plan(result.plan)
    macro_tbl = Table(title="Macro totals (actual vs target)")
    macro_tbl.add_column("Nutrient")
    macro_tbl.add_column("Actual", justify="right")
    macro_tbl.add_column("Target", justify="right")
    for label, nutrient, target in (
        ("kcal", NutrientName.ENERGY_KCAL, result.targets.energy_kcal),
        ("protein g", NutrientName.PROTEIN_G, result.targets.protein_g),
        ("carbs g", NutrientName.CARBS_G, result.targets.carbs_g),
        ("fat g", NutrientName.FAT_G, result.targets.fat_g),
    ):
        macro_tbl.add_row(label, f"{totals.get(nutrient):.0f}", f"{target:.0f}")
    console.print(macro_tbl)

    if result.citations:
        console.print(f"\n[bold]Citations[/bold]: {len(result.citations)} chunks retrieved.")

    if _verbose:
        _render_telemetry(result.telemetry)


def _render_telemetry(telemetry: object) -> None:
    """Print per-tool call counts and LLM request/token usage (--verbose only)."""
    from dietary_advisor.telemetry import RunTelemetry  # local import to avoid cycles

    assert isinstance(telemetry, RunTelemetry)
    debug_tbl = Table(title="Debug: LLM & tool usage")
    debug_tbl.add_column("Tool")
    debug_tbl.add_column("Calls", justify="right")
    for tool_name, count in sorted(telemetry.tool_calls.items()):
        debug_tbl.add_row(tool_name, str(count))
    debug_tbl.add_row("[dim]total tool calls[/dim]", str(telemetry.total_tool_calls))
    console.print(debug_tbl)
    console.print(
        f"[dim]LLM requests: {telemetry.requests} | "
        f"input tokens: {telemetry.input_tokens} | "
        f"output tokens: {telemetry.output_tokens} | "
        f"total tokens: {telemetry.total_tokens}[/dim]",
    )


# ---------- evaluate ----------------------------------------------------------------------------


@app.command()
def evaluate(
    levels: str = typer.Option("1,2,3", "--levels", help="Comma-separated complexity levels."),
    output: Path = typer.Option(Path("evaluation/reports/ablation.csv"), "--output"),
    repeats: int = typer.Option(1, "--repeats", help="Repeat each (variant,profile,query) N times."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip G-Eval soft-preference scoring."),
) -> None:
    """Run the leave-one-out ablation grid (full system + one config per disabled module)."""
    # Local imports keep `cli --help` fast even when matplotlib isn't built.
    from evaluation.ablation import leave_one_out_variants
    from evaluation.report import write_markdown_summary, write_plots
    from evaluation.runner import run_ablation_grid

    level_ids = [int(s) for s in levels.split(",") if s.strip()]

    df = asyncio.run(
        run_ablation_grid(
            variants=leave_one_out_variants(),
            levels=level_ids,
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
