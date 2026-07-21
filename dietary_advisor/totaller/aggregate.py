"""Totaller: deterministic aggregator over a `MealPlan`.

This is the symbolic shield against the LLM's well-documented arithmetic
hallucinations (NutriGen reports MAE of 10.45 % for vanilla DeepSeek V3
on caloric sums - see Kwerenda Literatury sec. "Ewaluacja przez CSR").

The function uses :class:`fractions.Fraction` internally to avoid any
floating-point creep when summing many small portions, then snaps to a
0.01 grid before returning floats.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from dietary_advisor.planning.meal_plan import MealNutrientTotals, MealPlan, NutrientTotals, Portion
from dietary_advisor.totaller.nutrition import NutrientName

log = logging.getLogger(__name__)

# Round all returned amounts to 2 decimals to keep CLI/eval tables readable.
_PRECISION_DIGITS = 2


def total_portion(portion: Portion) -> dict[NutrientName, Fraction]:
    """Exact per-nutrient totals for a single portion.

    Per-100g entries are scaled by `grams / 100` using Fractions so the sum of
    N portions is bit-exact regardless of order.
    """
    scale = Fraction(portion.grams).limit_denominator(10_000_000) / Fraction(100)
    out: dict[NutrientName, Fraction] = {}
    for nutrient, amount_per_100g in portion.food.nutrients_per_100g.items():
        out[nutrient] = Fraction(amount_per_100g).limit_denominator(10_000_000) * scale
    return out


def _round(acc: dict[NutrientName, Fraction]) -> dict[NutrientName, float]:
    return {nutrient: round(float(value), _PRECISION_DIGITS) for nutrient, value in acc.items()}


def total_meal_plan(plan: MealPlan) -> NutrientTotals:
    """Sum every portion in the plan into per-nutrient totals, overall and per meal.

    Per-meal sums are accumulated in exact `Fraction`s and folded into the
    overall total before either is rounded, so the overall total is the exact
    sum rather than a sum of already-rounded meal totals.
    """
    overall: dict[NutrientName, Fraction] = {}
    per_meal: list[MealNutrientTotals] = []
    for meal in plan.meals:
        meal_acc: dict[NutrientName, Fraction] = {}
        for portion in meal.portions:
            for nutrient, value in total_portion(portion).items():
                meal_acc[nutrient] = meal_acc.get(nutrient, Fraction(0)) + value
        for nutrient, value in meal_acc.items():
            overall[nutrient] = overall.get(nutrient, Fraction(0)) + value
        per_meal.append(MealNutrientTotals(kind=meal.kind, name=meal.name, totals=_round(meal_acc)))

    rounded = _round(overall)
    log.info(
        "totaller.total_meal_plan(%d meal(s)) -> kcal=%.0f protein=%.1fg carbs=%.1fg fat=%.1fg",
        len(plan.meals),
        rounded.get(NutrientName.ENERGY_KCAL, 0.0),
        rounded.get(NutrientName.PROTEIN_G, 0.0),
        rounded.get(NutrientName.CARBS_G, 0.0),
        rounded.get(NutrientName.FAT_G, 0.0),
    )
    return NutrientTotals(totals=rounded, per_meal=per_meal)
