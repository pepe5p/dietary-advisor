"""Tests for the ground-truth (evaluation-only) validator + executable hard rules."""

from __future__ import annotations

from dietary_advisor.planning.meal_plan import (
    Meal,
    MealPlan,
    Portion,
)
from dietary_advisor.totaller.nutrition import FoodItem, NutrientName
from evaluation.constraints import HardConstraint
from evaluation.validation.validator import validate_meal_plan
from tests.conftest import LONG_INSTRUCTIONS


def _plan_with(*portions: Portion, kind: str = "lunch") -> MealPlan:
    return MealPlan(
        user_id="x",
        meals=[Meal(kind=kind, name="r", portions=list(portions), recipe=LONG_INSTRUCTIONS)],
    )


def test_no_constraints_means_satisfied(rice_food: FoodItem) -> None:
    plan = _plan_with(Portion(food=rice_food, grams=200))
    report = validate_meal_plan(plan, [])
    assert report.hard_satisfied
    assert not report.violations
    assert report.hsr == 1.0


def test_allergen_and_diet_constraints_are_skipped_by_validator(
    rice_food: FoodItem,
    peanut_food: FoodItem,
) -> None:
    # Allergen/diet kinds are LLM-judged; the code validator ignores them.
    plan = _plan_with(
        Portion(food=rice_food, grams=200),
        Portion(food=peanut_food, grams=20),
    )
    report = validate_meal_plan(
        plan,
        [HardConstraint.allergen("peanuts"), HardConstraint.diet("vegan")],
    )
    assert report.hard_satisfied
    assert not report.violations


def test_max_nutrient_violation(salty_food: FoodItem) -> None:
    plan = _plan_with(Portion(food=salty_food, grams=200))  # 3000 mg sodium
    report = validate_meal_plan(plan, [HardConstraint.max_nutrient(NutrientName.SODIUM_MG, 2000.0)])
    assert not report.hard_satisfied
    assert report.totals[NutrientName.SODIUM_MG] > 2000.0


def test_min_nutrient_violation(rice_food: FoodItem) -> None:
    # 200g rice = 0.8 g fiber; require >= 25 g/day -> violation expected.
    plan = _plan_with(Portion(food=rice_food, grams=200))
    report = validate_meal_plan(
        plan,
        [HardConstraint.min_nutrient(NutrientName.FIBER_G, 25.0)],
    )
    assert not report.hard_satisfied


def test_ingredient_exclusion(salty_food: FoodItem) -> None:
    plan = _plan_with(Portion(food=salty_food, grams=50))
    report = validate_meal_plan(
        plan,
        [HardConstraint(kind="ingredient_exclusion", target="ham")],
    )
    assert not report.hard_satisfied


def test_meal_count_violation_when_too_few(rice_food: FoodItem) -> None:
    plan = _plan_with(Portion(food=rice_food, grams=200))  # exactly 1 meal
    report = validate_meal_plan(plan, [HardConstraint.meal_count(3)])
    assert not report.hard_satisfied
    assert "3" in report.violations[0].detail


def test_meal_count_satisfied_on_exact_match(rice_food: FoodItem) -> None:
    plan = _plan_with(Portion(food=rice_food, grams=200))  # exactly 1 meal
    report = validate_meal_plan(plan, [HardConstraint.meal_count(1)])
    assert report.hard_satisfied
    assert not report.violations


def test_hsr_partial_pass(rice_food: FoodItem, peanut_food: FoodItem) -> None:
    plan = _plan_with(
        Portion(food=rice_food, grams=200),
        Portion(food=peanut_food, grams=10),
    )
    constraints = [
        HardConstraint(kind="ingredient_exclusion", target="peanut"),  # violated
        HardConstraint.meal_count(1),  # satisfied
        HardConstraint.min_nutrient(NutrientName.FIBER_G, 0.1),  # satisfied
        HardConstraint.max_nutrient(NutrientName.SODIUM_MG, 10_000.0),  # satisfied
    ]
    report = validate_meal_plan(plan, constraints)
    assert report.hsr == 0.75
