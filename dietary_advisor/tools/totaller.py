"""Totaller: deterministic aggregator over a `MealPlan`.

This is the symbolic shield against the LLM's well-documented arithmetic
hallucinations (NutriGen reports MAE of 10.45 % for vanilla DeepSeek V3
on caloric sums - see Kwerenda Literatury sec. "Ewaluacja przez CSR").

The function uses :class:`fractions.Fraction` internally to avoid any
floating-point creep when summing many small portions, then snaps to a
0.01 grid before returning floats.
"""

from __future__ import annotations

from fractions import Fraction

from dietary_advisor.schemas.meal_plan import MealPlan, NutrientTotals, Portion
from dietary_advisor.schemas.nutrition import NutrientName

# Round all returned amounts to 2 decimals to keep CLI/eval tables readable.
_PRECISION_DIGITS = 2


def total_portion(portion: Portion) -> dict[NutrientName, Fraction]:
    """Exact per-nutrient totals for a single portion.

    USDA per-100g entries are scaled by `grams / 100` using Fractions so the
    sum of N portions is bit-exact regardless of order.
    """
    scale = Fraction(portion.grams).limit_denominator(10_000_000) / Fraction(100)
    out: dict[NutrientName, Fraction] = {}
    for nutrient, amount_per_100g in portion.food.nutrients_per_100g.items():
        out[nutrient] = Fraction(amount_per_100g).limit_denominator(10_000_000) * scale
    return out


def total_meal_plan(plan: MealPlan) -> NutrientTotals:
    """Sum every portion in the plan into per-nutrient totals (canonical units)."""
    acc: dict[NutrientName, Fraction] = {}
    for meal in plan.meals:
        for portion in meal.recipe.portions:
            for nutrient, value in total_portion(portion).items():
                acc[nutrient] = acc.get(nutrient, Fraction(0)) + value

    rounded: dict[NutrientName, float] = {
        nutrient: round(float(value), _PRECISION_DIGITS) for nutrient, value in acc.items()
    }
    return NutrientTotals(totals=rounded)


def total_meal_plan_dict(plan: MealPlan) -> dict[str, float]:
    """Convenience wrapper returning a plain `{nutrient_name: amount}` dict.

    This is the shape exposed to the LLM through the agent tool layer.
    """
    return {n.value: v for n, v in total_meal_plan(plan).totals.items()}
