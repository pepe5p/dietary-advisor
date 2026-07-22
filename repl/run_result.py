"""Render a hardcoded `PipelineResult` through the CLI's own `render_result`.

Lets display tweaks in `dietary_advisor/cli/rendering.py` be previewed from
the REPL without running the LLM pipeline or opening the food DB.
"""

from __future__ import annotations

from collections import Counter

from dietary_advisor.agents.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.cli.rendering import render_result
from dietary_advisor.planning.meal_plan import Citation, Meal, MealPlan, Portion
from dietary_advisor.planning.pipeline import PipelineResult
from dietary_advisor.planning.shopping_list import build_shopping_list
from dietary_advisor.telemetry import RunTelemetry
from dietary_advisor.totaller.nutrition import FoodItem, MacroTargets, NutrientName
from repl.manual import print_manual

__all__ = ["print_run_result", "sample_result"]

_OATS = FoodItem(
    code="3017620422003",
    name="Rolled oats",
    quantity_g=500.0,
    nutrients_per_100g={
        NutrientName.ENERGY_KCAL: 380.0,
        NutrientName.PROTEIN_G: 13.0,
        NutrientName.CARBS_G: 62.0,
        NutrientName.FAT_G: 7.0,
        NutrientName.SATURATED_FAT_G: 1.2,
        NutrientName.FIBER_G: 10.0,
        NutrientName.SUGAR_G: 1.0,
        NutrientName.SALT_G: 0.01,
        NutrientName.SODIUM_MG: 2.0,
        NutrientName.IRON_MG: 4.3,
    },
)
_BLUEBERRIES = FoodItem(
    code="usda:173946",
    name="Blueberries, raw",
    nutrients_per_100g={
        NutrientName.ENERGY_KCAL: 57.0,
        NutrientName.PROTEIN_G: 0.7,
        NutrientName.CARBS_G: 14.5,
        NutrientName.FAT_G: 0.3,
        NutrientName.FIBER_G: 2.4,
        NutrientName.SUGAR_G: 10.0,
        NutrientName.VITAMIN_C_MG: 9.7,
        NutrientName.POTASSIUM_MG: 77.0,
    },
)
_CHICKEN_BREAST = FoodItem(
    code="usda:171077",
    name="Chicken breast, grilled",
    nutrients_per_100g={
        NutrientName.ENERGY_KCAL: 165.0,
        NutrientName.PROTEIN_G: 31.0,
        NutrientName.CARBS_G: 0.0,
        NutrientName.FAT_G: 3.6,
        NutrientName.SATURATED_FAT_G: 1.0,
        NutrientName.SODIUM_MG: 74.0,
        NutrientName.CHOLESTEROL_MG: 85.0,
    },
)
_QUINOA = FoodItem(
    code="8712345678901",
    name="Cooked quinoa",
    quantity_g=250.0,
    nutrients_per_100g={
        NutrientName.ENERGY_KCAL: 120.0,
        NutrientName.PROTEIN_G: 4.4,
        NutrientName.CARBS_G: 21.3,
        NutrientName.FAT_G: 1.9,
        NutrientName.FIBER_G: 2.8,
        NutrientName.SODIUM_MG: 7.0,
        NutrientName.IRON_MG: 1.5,
    },
)
_SALMON = FoodItem(
    code="20123456",
    name="Baked salmon fillet",
    quantity_g=400.0,
    nutrients_per_100g={
        NutrientName.ENERGY_KCAL: 208.0,
        NutrientName.PROTEIN_G: 20.4,
        NutrientName.CARBS_G: 0.0,
        NutrientName.FAT_G: 13.4,
        NutrientName.SATURATED_FAT_G: 3.1,
        NutrientName.SALT_G: 0.8,
        NutrientName.SODIUM_MG: 59.0,
        NutrientName.VITAMIN_D_UG: 11.0,
        NutrientName.CHOLESTEROL_MG: 55.0,
    },
)
_BROCCOLI = FoodItem(
    code="usda:170379",
    name="Steamed broccoli",
    nutrients_per_100g={
        NutrientName.ENERGY_KCAL: 35.0,
        NutrientName.PROTEIN_G: 2.4,
        NutrientName.CARBS_G: 7.2,
        NutrientName.FAT_G: 0.4,
        NutrientName.FIBER_G: 3.3,
        NutrientName.VITAMIN_C_MG: 64.9,
        NutrientName.CALCIUM_MG: 47.0,
        NutrientName.POTASSIUM_MG: 293.0,
    },
)

_BREAKFAST_RECIPE = (
    "Combine the rolled oats with 240ml of water or milk in a small saucepan. Bring to a gentle "
    "simmer over medium heat, stirring occasionally, and cook for about 5 minutes until thickened. "
    "Remove from heat, stir in the blueberries, and let sit for a minute before serving warm."
)
_LUNCH_RECIPE = (
    "Season the chicken breast with salt, pepper, and a squeeze of lemon, then grill over "
    "medium-high heat for 6-7 minutes per side until cooked through. Rinse the quinoa and simmer "
    "in salted water for 15 minutes until fluffy. Steam the broccoli for 4 minutes until "
    "tender-crisp, then plate everything together."
)
_DINNER_RECIPE = (
    "Pat the salmon fillet dry and season with salt, pepper, and dill. Bake in a preheated oven "
    "at 200C for 12-15 minutes until just cooked through and flaky. Meanwhile, simmer the quinoa "
    "in stock for 15 minutes and steam the broccoli for 4 minutes; plate the salmon over the "
    "quinoa with the broccoli on the side."
)

_RATIONALE = (
    "This plan favours high-fibre wholegrains and a variety of vegetables at each meal, with lean "
    "protein (chicken, salmon) spread across lunch and dinner to help meet the protein target "
    "without excess saturated fat."
)


def _to_portion_ref(portion: Portion) -> PortionRef:
    assert portion.food.code is not None, "sample FoodItems always carry a code"
    return PortionRef(code=portion.food.code, name=portion.food.name, grams=portion.grams)


def _to_agent_meal(meal: Meal) -> AgentMeal:
    """Derive the reference-only `AgentMeal` for `meal` so both plans stay in sync."""
    return AgentMeal(
        kind=meal.kind,
        recipe=AgentRecipe(
            name=meal.name,
            portions=[_to_portion_ref(p) for p in meal.portions],
            instructions=meal.recipe,
        ),
    )


def sample_result() -> PipelineResult:
    """Build a hardcoded `PipelineResult` covering every branch of `_render_result`.

    Reuses the same ingredient (`_QUINOA`, `_BROCCOLI`) across lunch and dinner so the shopping
    list's cross-meal aggregation is also exercised, and mixes an Open Food Facts code with
    `usda:`-prefixed codes so both `Source` column branches render.
    """
    meals = [
        Meal(
            kind="breakfast",
            name="Blueberry oatmeal",
            portions=[Portion(food=_OATS, grams=60), Portion(food=_BLUEBERRIES, grams=80)],
            recipe=_BREAKFAST_RECIPE,
        ),
        Meal(
            kind="lunch",
            name="Grilled chicken quinoa bowl",
            portions=[
                Portion(food=_CHICKEN_BREAST, grams=150),
                Portion(food=_QUINOA, grams=150),
                Portion(food=_BROCCOLI, grams=100),
            ],
            recipe=_LUNCH_RECIPE,
        ),
        Meal(
            kind="dinner",
            name="Baked salmon with quinoa and broccoli",
            portions=[
                Portion(food=_SALMON, grams=150),
                Portion(food=_QUINOA, grams=100),
                Portion(food=_BROCCOLI, grams=100),
            ],
            recipe=_DINNER_RECIPE,
        ),
    ]
    citations = [
        Citation(
            source="NICE_NG28",
            page=12,
            snippet="Encourage a diet high in fibre and wholegrain cereals as part of a balanced diet.",
        ),
        Citation(
            source="USDA_DGA_2020",
            snippet="Adults should consume a variety of vegetables from all subgroups each week.",
        ),
    ]
    plan = MealPlan(user_id="sample-user", meals=meals, rationale=_RATIONALE, citations=citations)
    agent_plan = AgentMealPlan(
        user_id=plan.user_id,
        meals=[_to_agent_meal(m) for m in plan.meals],
        rationale=plan.rationale,
        citations=plan.citations,
    )
    return PipelineResult(
        plan=plan,
        agent_plan=agent_plan,
        targets=MacroTargets(energy_kcal=2100, protein_g=130, carbs_g=230, fat_g=70),
        citations=citations,
        iterations=1,
        variant="full",
        shopping_list=build_shopping_list(plan),
        telemetry=RunTelemetry(
            tool_calls=Counter({"lookup_food": 6, "lookup_foods": 2, "total_meal_plan": 3}),
            requests=4,
            input_tokens=5200,
            output_tokens=1800,
        ),
    )


def print_run_result(result: PipelineResult | None = None) -> None:
    """Render `result` (defaults to `sample_result()`) via the CLI's own `render_result`."""
    render_result(result if result is not None else sample_result())


_MANUAL: tuple[tuple[str, str], ...] = (
    ("sample_result()", "Build a hardcoded, DB-independent PipelineResult -> exercises every _render_result branch."),
    ("print_run_result(result=None)", "Render a PipelineResult (defaults to sample_result()) via cli.render_result."),
)

print_manual("run result helpers", _MANUAL)
