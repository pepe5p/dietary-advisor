"""Rich console rendering of a finished `PipelineResult`."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from dietary_advisor.food_db.usda_food_db import is_usda_code
from dietary_advisor.planning.meal_plan import Portion
from dietary_advisor.planning.pipeline import PipelineResult
from dietary_advisor.telemetry import RunTelemetry
from dietary_advisor.totaller.aggregate import total_meal_plan, total_portion
from dietary_advisor.totaller.nutrition import NutrientName

console = Console()


def _portion_macros(portion: Portion) -> tuple[float, float, float, float]:
    """Return (kcal, protein g, carbs g, fat g) for a single portion."""
    amounts = total_portion(portion)
    return (
        float(amounts.get(NutrientName.ENERGY_KCAL, 0)),
        float(amounts.get(NutrientName.PROTEIN_G, 0)),
        float(amounts.get(NutrientName.CARBS_G, 0)),
        float(amounts.get(NutrientName.FAT_G, 0)),
    )


def render_result(result: PipelineResult, *, verbose: bool = False) -> None:
    """Pretty-print a `PipelineResult` to the console."""
    console.rule(f"Variant {result.variant} - reflection iterations: {result.iterations}")

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

    if result.plan.rationale:
        console.print(f"\n[bold]Rationale[/bold]: {result.plan.rationale}")

    plan_tbl = Table(title=f"MealPlan ({len(result.plan.meals)} meals)", show_lines=True)
    plan_tbl.add_column("Meal")
    plan_tbl.add_column("Meal Name")
    plan_tbl.add_column("Products")
    plan_tbl.add_column("Meal totals")
    plan_tbl.add_column("Recipe")
    for meal, meal_totals in zip(result.plan.meals, totals.per_meal, strict=True):
        lines = []
        for p in meal.portions:
            kcal, protein, carbs, fat = _portion_macros(p)
            lines.append(f"{p.food.name}: {p.grams:.0f}g — {kcal:.0f} kcal, P{protein:.1f} C{carbs:.1f} F{fat:.1f}")
        products = "\n".join(lines)
        meal_totals_cell = (
            f"{meal_totals.get(NutrientName.ENERGY_KCAL):.0f} kcal\n"
            f"P{meal_totals.get(NutrientName.PROTEIN_G):.1f} "
            f"C{meal_totals.get(NutrientName.CARBS_G):.1f} "
            f"F{meal_totals.get(NutrientName.FAT_G):.1f}"
        )
        plan_tbl.add_row(meal.kind, meal.name, products, meal_totals_cell, meal.recipe)
    console.print(plan_tbl)

    if result.shopping_list.items:
        shop_tbl = Table(title="Shopping list")
        shop_tbl.add_column("Ingredient")
        shop_tbl.add_column("Source")
        shop_tbl.add_column("DB ID")
        shop_tbl.add_column("Total (g)", justify="right")
        shop_tbl.add_column("kcal", justify="right")
        shop_tbl.add_column("Protein g", justify="right")
        shop_tbl.add_column("Carbs g", justify="right")
        shop_tbl.add_column("Fat g", justify="right")
        for item in result.shopping_list.items:
            source = "[cyan]USDA[/cyan]" if item.code and is_usda_code(item.code) else "[green]OFF[/green]"
            shop_tbl.add_row(
                item.name,
                source,
                item.code or "-",
                f"{item.total_grams:.0f}",
                f"{item.energy_kcal:.0f}",
                f"{item.protein_g:.1f}",
                f"{item.carbs_g:.1f}",
                f"{item.fat_g:.1f}",
            )
        console.print(shop_tbl)

    if result.citations:
        cite_tbl = Table(title=f"Citations ({len(result.citations)} chunks retrieved)", show_lines=True)
        cite_tbl.add_column("Source")
        cite_tbl.add_column("Page")
        cite_tbl.add_column("Snippet")
        for c in result.citations:
            cite_tbl.add_row(c.source, str(c.page) if c.page is not None else "-", c.snippet)
        console.print(cite_tbl)

    if verbose:
        _render_telemetry(result.telemetry)


def _render_telemetry(telemetry: RunTelemetry) -> None:
    """Print per-tool call counts and LLM request/token usage (--verbose only)."""
    debug_tbl = Table(title="Debug: LLM & tool usage")
    debug_tbl.add_column("Tool")
    debug_tbl.add_column("Calls", justify="right")
    for tool_name, count in sorted(telemetry.tool_calls.items()):
        debug_tbl.add_row(tool_name, str(count))
    debug_tbl.add_row("[dim]total tool calls[/dim]", str(telemetry.total_tool_calls))
    console.print(debug_tbl)
    input_cell = f"input tokens: {telemetry.input_tokens}"
    if telemetry.cache_read_tokens:
        input_cell += f" (cached: {telemetry.cache_read_tokens})"
    output_cell = f"output tokens: {telemetry.output_tokens}"
    if telemetry.reasoning_tokens:
        output_cell += f" (reasoning: {telemetry.reasoning_tokens})"
    cache_write_cell = f" | cache write tokens: {telemetry.cache_write_tokens}" if telemetry.cache_write_tokens else ""
    console.print(
        f"[dim]LLM requests: {telemetry.requests} | "
        f"{input_cell} | "
        f"{output_cell}{cache_write_cell} | "
        f"total tokens: {telemetry.total_tokens}[/dim]",
    )
