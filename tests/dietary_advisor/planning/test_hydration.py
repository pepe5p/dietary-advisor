"""Unit tests for hydration: mapping read models and resolving agent meal plans."""

from __future__ import annotations

import pytest

from dietary_advisor.agents.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.food_db import OFFItem, USDAItem
from dietary_advisor.food_db.errors import MultipleUnknownFoodCodesError, UnknownFoodCodeError
from dietary_advisor.planning.hydration import hydrate_meal_plan, to_food_item
from dietary_advisor.totaller.nutrition import NutrientName
from tests.conftest import LONG_INSTRUCTIONS, LONG_RATIONALE


class _FakeFoodDb:
    def __init__(self, known_codes: set[str]) -> None:
        self._known_codes = known_codes

    def get_food(self, code: str) -> OFFItem:
        if code not in self._known_codes:
            raise UnknownFoodCodeError(code, f"unknown code: {code!r}")
        return OFFItem(code=code, product_name=f"Food {code}", energy_kcal_in_100g=100.0)


def test_off_item_maps_to_food_item() -> None:
    off = OFFItem(
        code="123",
        product_name="Mozzarella",
        energy_kcal_in_100g=250.0,
        brands="Acme",
        product_quantity=200.0,
    )
    food = to_food_item(off)
    assert food.code == "off:123"
    assert food.name == "Mozzarella"
    assert food.nutrients_per_100g == {NutrientName.ENERGY_KCAL: 250.0}
    assert food.quantity_g == 200.0


def test_usda_item_maps_to_food_item() -> None:
    usda = USDAItem(
        fdc_id=456,
        description="Cheddar cheese",
        energy_kcal_in_100g=400.0,
        category="Dairy",
    )
    food = to_food_item(usda)
    assert food.code == "usda:456"
    assert food.name == "Cheddar cheese"
    assert food.nutrients_per_100g == {NutrientName.ENERGY_KCAL: 400.0}
    assert food.quantity_g is None


def test_hydrate_meal_plan_raises_all_unknown_codes() -> None:
    plan = AgentMealPlan(
        user_id="t",
        meals=[
            AgentMeal(
                kind="breakfast",
                recipe=AgentRecipe(
                    name="Breakfast",
                    portions=[PortionRef(code="bad-a", name="Bad A", grams=100.0)],
                    instructions=LONG_INSTRUCTIONS,
                ),
            ),
            AgentMeal(
                kind="lunch",
                recipe=AgentRecipe(
                    name="Lunch",
                    portions=[PortionRef(code="bad-b", name="Bad B", grams=100.0)],
                    instructions=LONG_INSTRUCTIONS,
                ),
            ),
        ],
        rationale=LONG_RATIONALE,
    )
    with pytest.raises(MultipleUnknownFoodCodesError) as exc_info:
        hydrate_meal_plan(plan, _FakeFoodDb(set()))  # type: ignore[arg-type]
    assert exc_info.value.codes == ["bad-a", "bad-b"]
