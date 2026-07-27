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
from dietary_advisor.totaller.nutrition import canonical_unit, nutrient_label, NutrientName

log = logging.getLogger(__name__)

# Round all returned amounts to 2 decimals to keep CLI/eval tables readable.
_PRECISION_DIGITS = 2
_MAX_NAMED_FOODS = 3


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


def _format_missing_foods(names: list[str]) -> str:
    if len(names) <= _MAX_NAMED_FOODS:
        return ", ".join(names)
    rest = len(names) - _MAX_NAMED_FOODS
    return f"{', '.join(names[:_MAX_NAMED_FOODS])} and {rest} more"


def _coverage_warnings(plan: MealPlan, rounded_totals: dict[NutrientName, float]) -> list[str]:
    """Flag nutrients whose summed totals omit foods with no DB value for that nutrient.

    A key absent from ``nutrients_per_100g`` means unknown; a present ``0.0`` is a real zero.
    """
    portions = [portion for meal in plan.meals for portion in meal.portions]
    if not portions:
        return []

    total_grams = sum(Fraction(portion.grams).limit_denominator(10_000_000) for portion in portions)
    if total_grams == 0:
        return []

    n_portions = len(portions)
    missing_grams: dict[NutrientName, Fraction] = {}
    missing_count: dict[NutrientName, int] = {}
    missing_foods: dict[NutrientName, list[str]] = {}

    for portion in portions:
        grams = Fraction(portion.grams).limit_denominator(10_000_000)
        per_100g = portion.food.nutrients_per_100g
        for nutrient in NutrientName:
            if nutrient in per_100g:
                continue
            missing_grams[nutrient] = missing_grams.get(nutrient, Fraction(0)) + grams
            missing_count[nutrient] = missing_count.get(nutrient, 0) + 1
            names = missing_foods.setdefault(nutrient, [])
            if portion.food.name not in names:
                names.append(portion.food.name)

    gaps = [nutrient for nutrient in NutrientName if missing_count.get(nutrient, 0) > 0]
    if not gaps:
        return []

    gaps.sort(key=lambda nutrient: (-float(missing_grams[nutrient] / total_grams), nutrient.value))

    warnings: list[str] = []
    total_round = round(float(total_grams))
    for nutrient in gaps:
        missing_g = missing_grams[nutrient]
        count = missing_count[nutrient]
        missing_round = round(float(missing_g))
        pct = round(float(missing_g / total_grams * 100))
        label = nutrient_label(nutrient)
        unit = canonical_unit(nutrient)
        total_val = rounded_totals.get(nutrient, 0.0)
        mass_clause = f"{missing_round} g of {total_round} g ({pct}% of plan mass)"

        if count == n_portions:
            warnings.append(
                f"{label}: none of the {n_portions} foods has {label} data - "
                f"{mass_clause}, so the {total_val:g} {unit} total is unknown, not a real zero",
            )
        else:
            foods = _format_missing_foods(missing_foods[nutrient])
            warnings.append(
                f"{label}: {count} of {n_portions} foods have no {label} data - "
                f"{mass_clause}, so the {total_val:g} {unit} total is understated ({foods})",
            )
    return warnings


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
    warnings = _coverage_warnings(plan, rounded)
    log.info(
        "totaller.total_meal_plan(%d meal(s)) -> kcal=%.0f protein=%.1fg carbs=%.1fg fat=%.1fg warnings=%d",
        len(plan.meals),
        rounded.get(NutrientName.ENERGY_KCAL, 0.0),
        rounded.get(NutrientName.PROTEIN_G, 0.0),
        rounded.get(NutrientName.CARBS_G, 0.0),
        rounded.get(NutrientName.FAT_G, 0.0),
        len(warnings),
    )
    return NutrientTotals(totals=rounded, per_meal=per_meal, warnings=warnings)
