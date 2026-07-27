"""Totaller arithmetic precision tests."""

from __future__ import annotations

from fractions import Fraction

import pytest

from dietary_advisor.planning.meal_plan import (
    Meal,
    MealPlan,
    Portion,
)
from dietary_advisor.totaller.aggregate import (
    total_meal_plan,
    total_portion,
)
from dietary_advisor.totaller.nutrition import FoodItem, NutrientName
from tests.conftest import LONG_INSTRUCTIONS


def test_total_portion_is_exact() -> None:
    food = FoodItem(name="thing", nutrients_per_100g={NutrientName.ENERGY_KCAL: 333.333})
    p = Portion(food=food, grams=37.5)
    out = total_portion(p)
    assert out[NutrientName.ENERGY_KCAL] == Fraction(333333, 1000) * Fraction(375, 1000)


def test_totaller_aggregates_multiple_meals(chicken_food: FoodItem, rice_food: FoodItem) -> None:
    plan = MealPlan(
        user_id="x",
        meals=[
            Meal(
                kind="lunch",
                name="r",
                portions=[
                    Portion(food=chicken_food, grams=150),
                    Portion(food=rice_food, grams=200),
                ],
                recipe=LONG_INSTRUCTIONS,
            ),
            Meal(
                kind="dinner",
                name="r2",
                portions=[
                    Portion(food=chicken_food, grams=100),
                ],
                recipe=LONG_INSTRUCTIONS,
            ),
        ],
    )
    result = total_meal_plan(plan)
    totals = result.totals
    # 250g chicken at 165 kcal/100g + 200g rice at 130 kcal/100g
    expected_kcal = 250 * 165 / 100 + 200 * 130 / 100
    assert totals[NutrientName.ENERGY_KCAL] == pytest.approx(expected_kcal, abs=0.05)
    expected_protein = 250 * 31 / 100 + 200 * 2.7 / 100
    assert totals[NutrientName.PROTEIN_G] == pytest.approx(expected_protein, abs=0.05)

    # Per-meal breakdown, ordered like plan.meals: lunch (chicken+rice), dinner (chicken only).
    assert [m.kind for m in result.per_meal] == ["lunch", "dinner"]
    lunch, dinner = result.per_meal
    lunch_kcal = 150 * 165 / 100 + 200 * 130 / 100
    assert lunch.totals[NutrientName.ENERGY_KCAL] == pytest.approx(lunch_kcal, abs=0.05)
    dinner_kcal = 100 * 165 / 100
    assert dinner.totals[NutrientName.ENERGY_KCAL] == pytest.approx(dinner_kcal, abs=0.05)


def test_totaller_handles_many_small_portions() -> None:
    """Sum of 100 1-gram portions should equal 1 gram per macro."""
    food = FoodItem(name="micro", nutrients_per_100g={NutrientName.PROTEIN_G: 100.0})
    portions = [Portion(food=food, grams=1.0) for _ in range(100)]
    plan = MealPlan(
        user_id="x",
        meals=[Meal(kind="snack", name="r", portions=portions, recipe=LONG_INSTRUCTIONS)],
    )
    totals = total_meal_plan(plan).totals
    assert totals[NutrientName.PROTEIN_G] == pytest.approx(100.0)


def test_coverage_warnings_empty_when_full_coverage() -> None:
    food = FoodItem(name="complete", nutrients_per_100g=dict.fromkeys(NutrientName, 0.0))
    plan = MealPlan(
        user_id="x",
        meals=[Meal(kind="lunch", name="r", portions=[Portion(food=food, grams=100)], recipe=LONG_INSTRUCTIONS)],
    )
    assert total_meal_plan(plan).warnings == []


def test_coverage_warnings_skip_explicit_zero(chicken_food: FoodItem) -> None:
    meal = Meal(
        kind="lunch",
        name="r",
        portions=[Portion(food=chicken_food, grams=150)],
        recipe=LONG_INSTRUCTIONS,
    )
    plan = MealPlan(user_id="x", meals=[meal])
    warnings = total_meal_plan(plan).warnings
    assert not any("carbs" in warning for warning in warnings)


def test_coverage_warnings_partial_gap(chicken_food: FoodItem, rice_food: FoodItem) -> None:
    plan = MealPlan(
        user_id="x",
        meals=[
            Meal(
                kind="lunch",
                name="r",
                portions=[
                    Portion(food=chicken_food, grams=150),
                    Portion(food=rice_food, grams=200),
                ],
                recipe=LONG_INSTRUCTIONS,
            ),
        ],
    )
    warnings = total_meal_plan(plan).warnings
    fiber_warnings = [warning for warning in warnings if warning.startswith("fiber:")]
    assert len(fiber_warnings) == 1
    fiber = fiber_warnings[0]
    assert "1 of 2 foods have no fiber data" in fiber
    assert "150 g of 350 g (43% of plan mass)" in fiber
    assert "understated (Chicken breast)" in fiber


def test_coverage_warnings_total_gap(chicken_food: FoodItem) -> None:
    meal = Meal(
        kind="lunch",
        name="r",
        portions=[Portion(food=chicken_food, grams=150)],
        recipe=LONG_INSTRUCTIONS,
    )
    plan = MealPlan(user_id="x", meals=[meal])
    warnings = total_meal_plan(plan).warnings
    vitamin_d = next(w for w in warnings if w.startswith("vitamin d:"))
    assert "none of the 1 foods has vitamin d data" in vitamin_d
    assert "150 g of 150 g (100% of plan mass)" in vitamin_d
    assert "unknown, not a real zero" in vitamin_d


def test_coverage_warnings_sorted_by_missing_mass_share(
    chicken_food: FoodItem,
    rice_food: FoodItem,
    salty_food: FoodItem,
) -> None:
    plan = MealPlan(
        user_id="x",
        meals=[
            Meal(
                kind="lunch",
                name="r",
                portions=[
                    Portion(food=chicken_food, grams=100),
                    Portion(food=rice_food, grams=100),
                    Portion(food=salty_food, grams=300),
                ],
                recipe=LONG_INSTRUCTIONS,
            ),
        ],
    )
    warnings = total_meal_plan(plan).warnings
    shares = [int(w.split("(")[1].split("%")[0]) for w in warnings]
    assert shares == sorted(shares, reverse=True)
