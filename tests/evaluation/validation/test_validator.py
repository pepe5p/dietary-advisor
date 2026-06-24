"""Tests for the ground-truth (evaluation-only) validator + executable hard rules."""

from __future__ import annotations

from dietary_advisor.schemas.meal_plan import (
    Meal,
    MealKind,
    MealPlan,
    Portion,
    Recipe,
)
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName
from evaluation.constraints import HardConstraint
from evaluation.validation.validator import validate_meal_plan
from tests.conftest import LONG_INSTRUCTIONS


def _plan_with(*portions: Portion, kind: MealKind = MealKind.LUNCH) -> MealPlan:
    return MealPlan(
        user_id="x",
        meals=[Meal(kind=kind, recipe=Recipe(name="r", portions=list(portions), instructions=LONG_INSTRUCTIONS))],
    )


def test_no_constraints_means_satisfied(rice_food: FoodItem) -> None:
    plan = _plan_with(Portion(food=rice_food, grams=200))
    report = validate_meal_plan(plan, [])
    assert report.hard_satisfied
    assert not report.violations
    assert report.hsr == 1.0


def test_allergen_exclusion_flags_offender(rice_food: FoodItem, peanut_food: FoodItem) -> None:
    plan = _plan_with(
        Portion(food=rice_food, grams=200),
        Portion(food=peanut_food, grams=20),
    )
    report = validate_meal_plan(plan, [HardConstraint.allergen("peanuts")])
    assert not report.hard_satisfied
    assert any("peanut" in v.detail.lower() for v in report.violations)


def test_diet_pattern_requires_tag(chicken_food: FoodItem) -> None:
    # chicken_food has only "pescatarian" tag; vegan diet pattern should fail.
    plan = _plan_with(Portion(food=chicken_food, grams=100))
    report = validate_meal_plan(plan, [HardConstraint.diet("vegan")])
    assert not report.hard_satisfied
    assert "vegan" in report.violations[0].detail.lower()


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
        HardConstraint.allergen("peanuts"),  # violated
        HardConstraint.allergen("milk"),  # satisfied
        HardConstraint.allergen("eggs"),  # satisfied
        HardConstraint.allergen("crustaceans"),  # satisfied
    ]
    report = validate_meal_plan(plan, constraints)
    assert report.hsr == 0.75
