"""Tests for the deterministic shopping-list builder."""

from __future__ import annotations

import pytest

from dietary_advisor.schemas.meal_plan import Meal, MealKind, MealPlan, Portion, Recipe
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName
from dietary_advisor.tools.shopping_list import build_shopping_list
from tests.conftest import LONG_INSTRUCTIONS


def _plan(*meals: Meal) -> MealPlan:
    return MealPlan(user_id="x", meals=list(meals))


def test_shopping_list_aggregates_same_food_across_meals(
    chicken_food: FoodItem,
    rice_food: FoodItem,
) -> None:
    plan = _plan(
        Meal(
            kind=MealKind.LUNCH,
            recipe=Recipe(
                name="lunch",
                portions=[Portion(food=chicken_food, grams=150), Portion(food=rice_food, grams=200)],
                instructions=LONG_INSTRUCTIONS,
            ),
        ),
        Meal(
            kind=MealKind.DINNER,
            recipe=Recipe(
                name="dinner",
                portions=[Portion(food=chicken_food, grams=100)],
                instructions=LONG_INSTRUCTIONS,
            ),
        ),
    )
    items = {it.name: it for it in build_shopping_list(plan).items}
    chicken, rice = items["Chicken breast"], items["White rice cooked"]

    assert chicken.total_grams == 250.0
    assert chicken.energy_kcal == pytest.approx(250 * 165 / 100)
    assert chicken.protein_g == pytest.approx(250 * 31 / 100)
    assert chicken.carbs_g == pytest.approx(0.0)
    assert chicken.fat_g == pytest.approx(250 * 3.6 / 100)

    assert rice.total_grams == 200.0
    assert rice.energy_kcal == pytest.approx(200 * 130 / 100)
    assert rice.protein_g == pytest.approx(200 * 2.7 / 100)
    assert rice.carbs_g == pytest.approx(200 * 28.0 / 100)
    assert rice.fat_g == pytest.approx(200 * 0.3 / 100)


def test_shopping_list_is_sorted_by_name(chicken_food: FoodItem, rice_food: FoodItem) -> None:
    plan = _plan(
        Meal(
            kind=MealKind.LUNCH,
            recipe=Recipe(
                name="lunch",
                portions=[Portion(food=rice_food, grams=200), Portion(food=chicken_food, grams=150)],
                instructions=LONG_INSTRUCTIONS,
            ),
        ),
    )
    names = [it.name for it in build_shopping_list(plan).items]
    assert names == sorted(names, key=str.lower)


def test_shopping_list_keeps_distinct_codes() -> None:
    a = FoodItem(code="1", name="Rice", nutrients_per_100g={NutrientName.ENERGY_KCAL: 130.0})
    b = FoodItem(code="2", name="Rice", nutrients_per_100g={NutrientName.ENERGY_KCAL: 360.0})
    plan = _plan(
        Meal(
            kind=MealKind.LUNCH,
            recipe=Recipe(
                name="lunch",
                portions=[Portion(food=a, grams=100), Portion(food=b, grams=50)],
                instructions=LONG_INSTRUCTIONS,
            ),
        ),
    )
    items = build_shopping_list(plan).items
    assert {it.code for it in items} == {"1", "2"}
    assert all(it.total_grams > 0 for it in items)
    by_code = {it.code: it for it in items}
    assert by_code["1"].energy_kcal == pytest.approx(100 * 130 / 100)
    assert by_code["2"].energy_kcal == pytest.approx(50 * 360 / 100)


def test_shopping_list_marks_off_provenance() -> None:
    off_item = FoodItem(
        code="1",
        name="Rice",
        from_open_food_facts=True,
        nutrients_per_100g={NutrientName.ENERGY_KCAL: 130.0},
    )
    llm_item = FoodItem(name="Homemade stew", nutrients_per_100g={NutrientName.ENERGY_KCAL: 200.0})
    plan = _plan(
        Meal(
            kind=MealKind.LUNCH,
            recipe=Recipe(
                name="lunch",
                portions=[Portion(food=off_item, grams=100), Portion(food=llm_item, grams=100)],
                instructions=LONG_INSTRUCTIONS,
            ),
        ),
    )
    by_name = {it.name: it for it in build_shopping_list(plan).items}
    assert by_name["Rice"].from_open_food_facts is True
    assert by_name["Homemade stew"].from_open_food_facts is False
