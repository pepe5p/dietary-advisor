"""Tests for the production hydration step (AgentMealPlan -> MealPlan)."""

from __future__ import annotations

from dietary_advisor.food_db import FoodDb
from evaluation.validation.hydrate import hydrate_meal_plan
from tests.evaluation.conftest import agent_plan_single


def test_hydrate_resolves_code(food_db: FoodDb, any_code: str) -> None:
    plan = agent_plan_single(any_code)
    hydrated = hydrate_meal_plan(plan, food_db)
    assert len(hydrated.meals) == 1
    food = hydrated.meals[0].portions[0].food
    assert food.code == any_code
    assert food.nutrients_per_100g  # real product has resolved nutrients
