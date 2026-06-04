"""Evaluation test fixtures: mock food cache and sample agent plans."""

from __future__ import annotations

import pytest

from dietary_advisor.schemas.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.schemas.meal_plan import MealKind
from dietary_advisor.schemas.nutrition import FoodItem

# Stable fdc_ids for deterministic mock cache lookups.
FDC_RICE = 1001
FDC_CHICKEN = 1002
FDC_PEANUT = 1003
FDC_SALTY = 1004


class MockFoodLookup:
    """In-memory ``FoodLookup`` for evaluation unit tests."""

    def __init__(self, foods: dict[int, FoodItem]) -> None:
        self._foods = foods

    def get_food(self, fdc_id: int) -> FoodItem:
        if fdc_id not in self._foods:
            raise KeyError(f"unknown fdc_id: {fdc_id}")
        return self._foods[fdc_id]


@pytest.fixture()
def mock_foods(
    rice_food: FoodItem,
    chicken_food: FoodItem,
    peanut_food: FoodItem,
    salty_food: FoodItem,
) -> dict[int, FoodItem]:
    return {
        FDC_RICE: rice_food.model_copy(update={"fdc_id": FDC_RICE}),
        FDC_CHICKEN: chicken_food.model_copy(update={"fdc_id": FDC_CHICKEN}),
        FDC_PEANUT: peanut_food.model_copy(update={"fdc_id": FDC_PEANUT}),
        FDC_SALTY: salty_food.model_copy(update={"fdc_id": FDC_SALTY}),
    }


@pytest.fixture()
def mock_lookup(mock_foods: dict[int, FoodItem]) -> MockFoodLookup:
    return MockFoodLookup(mock_foods)


def agent_plan_rice_lunch(grams: float = 200.0, user_id: str = "test") -> AgentMealPlan:
    return AgentMealPlan(
        user_id=user_id,
        meals=[
            AgentMeal(
                kind=MealKind.LUNCH,
                recipe=AgentRecipe(
                    name="Rice bowl",
                    portions=[PortionRef(fdc_id=FDC_RICE, grams=grams)],
                    instructions="Cook rice.",
                ),
            ),
        ],
    )


def agent_plan_rice_and_chicken(
    rice_g: float = 300.0,
    chicken_g: float = 150.0,
    user_id: str = "test",
) -> AgentMealPlan:
    return AgentMealPlan(
        user_id=user_id,
        meals=[
            AgentMeal(
                kind=MealKind.LUNCH,
                recipe=AgentRecipe(
                    name="Plate",
                    portions=[
                        PortionRef(fdc_id=FDC_RICE, grams=rice_g),
                        PortionRef(fdc_id=FDC_CHICKEN, grams=chicken_g),
                    ],
                    instructions="Cook and serve.",
                ),
            ),
        ],
    )


def agent_plan_with_dinner(user_id: str = "test") -> AgentMealPlan:
    return AgentMealPlan(
        user_id=user_id,
        meals=[
            AgentMeal(
                kind=MealKind.BREAKFAST,
                recipe=AgentRecipe(
                    name="Breakfast",
                    portions=[PortionRef(fdc_id=FDC_RICE, grams=100.0)],
                ),
            ),
            AgentMeal(
                kind=MealKind.DINNER,
                recipe=AgentRecipe(
                    name="Dinner",
                    portions=[PortionRef(fdc_id=FDC_RICE, grams=200.0)],
                ),
            ),
        ],
    )
