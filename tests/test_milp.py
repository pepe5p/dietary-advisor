"""Tests for the MILP / LP diet optimiser."""

from __future__ import annotations

import pytest

from dietary_advisor.schemas.constraints import HardConstraint
from dietary_advisor.schemas.nutrition import FoodItem, MacroTargets, NutrientName
from dietary_advisor.tools.milp_optimizer import (
    build_meal_plan_from_optimization,
    optimize_portions,
)


@pytest.fixture
def small_food_set() -> list[FoodItem]:
    return [
        FoodItem(
            name="oats",
            nutrients_per_100g={
                NutrientName.ENERGY_KCAL: 380.0,
                NutrientName.PROTEIN_G: 13.0,
                NutrientName.CARBS_G: 67.0,
                NutrientName.FAT_G: 7.0,
                NutrientName.FIBER_G: 10.0,
            },
            tags=["vegan", "vegetarian"],
        ),
        FoodItem(
            name="chicken breast",
            nutrients_per_100g={
                NutrientName.ENERGY_KCAL: 165.0,
                NutrientName.PROTEIN_G: 31.0,
                NutrientName.CARBS_G: 0.0,
                NutrientName.FAT_G: 3.6,
            },
            tags=["pescatarian"],
        ),
        FoodItem(
            name="olive oil",
            nutrients_per_100g={
                NutrientName.ENERGY_KCAL: 884.0,
                NutrientName.PROTEIN_G: 0.0,
                NutrientName.CARBS_G: 0.0,
                NutrientName.FAT_G: 100.0,
            },
            tags=["vegan", "vegetarian", "pescatarian"],
        ),
    ]


def _targets() -> MacroTargets:
    return MacroTargets(energy_kcal=2000, protein_g=120, carbs_g=220, fat_g=65, fiber_g=25)


def test_optimizer_returns_optimal(small_food_set: list[FoodItem]) -> None:
    result = optimize_portions(small_food_set, _targets())
    assert result.status == "Optimal"
    assert result.totals[NutrientName.ENERGY_KCAL] == pytest.approx(2000.0, rel=0.05)


def test_optimizer_respects_max_nutrient(small_food_set: list[FoodItem]) -> None:
    constraints = [HardConstraint.max_nutrient(NutrientName.FAT_G, 40.0)]
    result = optimize_portions(small_food_set, _targets(), constraints)
    assert result.status == "Optimal"
    assert result.totals[NutrientName.FAT_G] <= 40.0 + 1e-3


def test_optimizer_filters_allergens(small_food_set: list[FoodItem]) -> None:
    # Tag oats as containing gluten and ban gluten.
    small_food_set[0].tags.append("contains:gluten")
    constraints = [HardConstraint.allergen("gluten")]
    result = optimize_portions(small_food_set, _targets(), constraints)
    # Oats should not be selected.
    assert result.grams.get("oats", 0.0) == 0.0


def test_optimizer_no_eligible_foods() -> None:
    foods = [FoodItem(name="x", tags=["contains:peanuts"])]
    result = optimize_portions(foods, _targets(), [HardConstraint.allergen("peanuts")])
    assert result.status == "NoEligibleFoods"


def test_build_meal_plan_from_result(small_food_set: list[FoodItem]) -> None:
    result = optimize_portions(small_food_set, _targets())
    plan = build_meal_plan_from_optimization("u", small_food_set, result)
    assert plan.meals
    assert all(p.grams > 0 for p in plan.meals[0].recipe.portions)
