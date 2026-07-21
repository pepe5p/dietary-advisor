"""Tests for the code-based hard rule helpers."""

from __future__ import annotations

from dietary_advisor.planning.meal_plan import Meal, MealPlan, Portion
from dietary_advisor.totaller.aggregate import total_meal_plan
from dietary_advisor.totaller.nutrition import FoodItem, NutrientName
from evaluation.constraints import HardConstraint
from evaluation.validation.rules import IngredientExclusionRule, rules_from_constraints
from tests.conftest import LONG_INSTRUCTIONS


def test_rules_from_constraints_skips_llm_judged_kinds() -> None:
    rules = rules_from_constraints(
        [
            HardConstraint.allergen("peanuts"),
            HardConstraint.diet("vegan"),
            HardConstraint.max_nutrient(NutrientName.SODIUM_MG, 2000.0),
        ]
    )
    assert len(rules) == 1
    assert rules[0].constraint.kind == "max_nutrient"


def test_ingredient_exclusion_matches_name() -> None:
    food = FoodItem(name="Salted ham", nutrients_per_100g={})
    plan = MealPlan(
        user_id="x",
        meals=[
            Meal(kind="lunch", name="r", portions=[Portion(food=food, grams=50)], recipe=LONG_INSTRUCTIONS),
        ],
    )
    rule = IngredientExclusionRule(HardConstraint(kind="ingredient_exclusion", target="ham"))
    violations = rule.check(plan, total_meal_plan(plan))
    assert len(violations) == 1
