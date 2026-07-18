"""Deterministic shopping-list builder.

The grocery list is computed symbolically from the finished `MealPlan` rather
than authored by the LLM: identical ingredients used across several meals must
sum to a single, exact quantity, which is precisely the kind of aggregation an
LLM gets wrong. Ingredients are keyed by product `code` when known (so two
different products that happen to share a display name stay distinct) and by
the lower-cased name otherwise.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from dietary_advisor.schemas.meal_plan import MealPlan, ShoppingList, ShoppingListItem
from dietary_advisor.schemas.nutrition import NutrientName

log = logging.getLogger(__name__)

_PRECISION_DIGITS = 1
_MACRO_PRECISION_DIGITS = 1

# Macro columns shown per shopping-list line, alongside total grams.
_MACRO_NUTRIENTS: tuple[tuple[str, NutrientName], ...] = (
    ("energy_kcal", NutrientName.ENERGY_KCAL),
    ("protein_g", NutrientName.PROTEIN_G),
    ("carbs_g", NutrientName.CARBS_G),
    ("fat_g", NutrientName.FAT_G),
)


def build_shopping_list(plan: MealPlan) -> ShoppingList:
    """Aggregate every portion across the plan into one line per ingredient."""
    totals: dict[tuple[str | None, str], Fraction] = {}
    names: dict[tuple[str | None, str], str] = {}
    # Keyed the same way as `totals`; each ingredient is assumed to carry the
    # same per-100g nutrients everywhere it's used (same key = same product).
    nutrients_per_100g: dict[tuple[str | None, str], dict[NutrientName, float]] = {}
    order: list[tuple[str | None, str]] = []
    for meal in plan.meals:
        for portion in meal.portions:
            food = portion.food
            key = (food.code, food.name.lower())
            if key not in totals:
                totals[key] = Fraction(0)
                names[key] = food.name
                nutrients_per_100g[key] = food.nutrients_per_100g
                order.append(key)
            totals[key] += Fraction(portion.grams).limit_denominator(10_000_000)

    items = []
    for key in order:
        grams = totals[key]
        scale = grams / Fraction(100)
        per_100g = nutrients_per_100g[key]
        macros = {
            field: round(
                float(Fraction(per_100g.get(nutrient, 0.0)).limit_denominator(10_000_000) * scale),
                _MACRO_PRECISION_DIGITS,
            )
            for field, nutrient in _MACRO_NUTRIENTS
        }
        items.append(
            ShoppingListItem(
                name=names[key],
                total_grams=round(float(grams), _PRECISION_DIGITS),
                code=key[0],
                **macros,
            ),
        )
    items.sort(key=lambda it: (it.name.lower(), it.code or ""))
    log.info("shopping_list.build_shopping_list(%d meal(s)) -> %d item(s)", len(plan.meals), len(items))
    return ShoppingList(items=items)
