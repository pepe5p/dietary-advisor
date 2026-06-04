"""Tests for plan hydration and pipeline adaptation."""

from __future__ import annotations

from dietary_advisor.schemas.meal_plan import Meal, MealKind, MealPlan, Portion, Recipe
from evaluation.validation.hydrate import hydrate_meal_plan, meal_plan_to_eval_plan
from tests.evaluation.conftest import agent_plan_rice_lunch, FDC_RICE


def test_hydrate_resolves_fdc_id(mock_lookup: object) -> None:
    plan = agent_plan_rice_lunch()
    hydrated = hydrate_meal_plan(plan, mock_lookup)  # type: ignore[arg-type]
    assert len(hydrated.meals) == 1
    assert hydrated.meals[0].recipe.portions[0].food.fdc_id == FDC_RICE


def test_meal_plan_to_eval_plan_extracts_fdc_id(rice_food: object) -> None:
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
                    portions=[Portion(food=food.model_copy(update={"fdc_id": FDC_RICE}), grams=100.0)],
                ),
            ),
        ],
    )
    conv = meal_plan_to_eval_plan(meal_plan)
    assert not conv.warnings
    assert conv.plan.meals[0].recipe.portions[0].fdc_id == FDC_RICE


def test_meal_plan_to_eval_plan_warns_missing_fdc_id(rice_food: object) -> None:
    from dietary_advisor.schemas.nutrition import FoodItem

    food = rice_food  # type: ignore[assignment]
    assert isinstance(food, FoodItem)
    meal_plan = MealPlan(
        user_id="x",
        meals=[
            Meal(
                kind=MealKind.LUNCH,
                recipe=Recipe(name="r", portions=[Portion(food=food, grams=100.0)]),
            ),
        ],
    )
    conv = meal_plan_to_eval_plan(meal_plan)
    assert conv.warnings
    assert not conv.plan.meals
