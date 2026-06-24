"""Tests for plan hydration and pipeline adaptation."""

from __future__ import annotations

from dietary_advisor.schemas.meal_plan import Meal, MealKind, MealPlan, Portion, Recipe
from dietary_advisor.tools.food_db import OffFoodDb
from evaluation.validation.hydrate import hydrate_meal_plan, meal_plan_to_eval_plan
from tests.conftest import LONG_INSTRUCTIONS
from tests.evaluation.conftest import agent_plan_single


def test_hydrate_resolves_code(off_db: OffFoodDb, any_code: str) -> None:
    plan = agent_plan_single(any_code)
    hydrated = hydrate_meal_plan(plan, off_db)
    assert len(hydrated.meals) == 1
    food = hydrated.meals[0].recipe.portions[0].food
    assert food.code == any_code
    assert food.nutrients_per_100g  # real product has resolved nutrients


def test_meal_plan_to_eval_plan_extracts_code(rice_food: object) -> None:
    from dietary_advisor.schemas.nutrition import FoodItem

    food = rice_food  # type: ignore[assignment]
    assert isinstance(food, FoodItem)
    meal_plan = MealPlan(
        user_id="x",
        meals=[
            Meal(
                kind=MealKind.LUNCH,
                recipe=Recipe(
                    name="r",
                    portions=[Portion(food=food.model_copy(update={"code": "123456"}), grams=100.0)],
                    instructions=LONG_INSTRUCTIONS,
                ),
            ),
        ],
    )
    conv = meal_plan_to_eval_plan(meal_plan)
    assert not conv.warnings
    assert conv.plan.meals[0].recipe.portions[0].code == "123456"


def test_meal_plan_to_eval_plan_warns_missing_code(rice_food: object) -> None:
    from dietary_advisor.schemas.nutrition import FoodItem

    food = rice_food  # type: ignore[assignment]
    assert isinstance(food, FoodItem)
    meal_plan = MealPlan(
        user_id="x",
        meals=[
            Meal(
                kind=MealKind.LUNCH,
                recipe=Recipe(name="r", portions=[Portion(food=food, grams=100.0)], instructions=LONG_INSTRUCTIONS),
            ),
        ],
    )
    conv = meal_plan_to_eval_plan(meal_plan)
    assert conv.warnings
    assert not conv.plan.meals
