"""Unit tests for hydration: mapping read models and resolving agent meal plans."""

from __future__ import annotations

import pytest

from dietary_advisor.agents.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.food_db import OFFItem, USDAItem
from dietary_advisor.food_db.errors import MultipleUnknownFoodCodesError, UnknownFoodCodeError
from dietary_advisor.planning.hydration import hydrate_meal_plan, to_food_item
from dietary_advisor.planning.meal_plan import Meal, MealPlan, Portion
from dietary_advisor.totaller.aggregate import total_meal_plan
from dietary_advisor.totaller.nutrition import FoodItem, NutrientName
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
    assert food.quantity_g is None
    # USDA is curated: a modelled nutrient the record omits is a real zero, not unknown.
    assert food.nutrients_per_100g[NutrientName.ENERGY_KCAL] == 400.0
    assert food.nutrients_per_100g[NutrientName.PROTEIN_G] == 0.0
    assert food.nutrients_per_100g[NutrientName.FIBER_G] == 0.0
    # Salt is the one nutrient USDA does not model as a column, so it stays absent.
    assert NutrientName.SALT_G not in food.nutrients_per_100g


def test_off_item_missing_nutrient_stays_unknown() -> None:
    off = OFFItem(code="123", product_name="Mystery", energy_kcal_in_100g=250.0)
    food = to_food_item(off)
    # Open Food Facts is crowd-sourced: an absent nutrient is unknown, not zero.
    assert food.nutrients_per_100g == {NutrientName.ENERGY_KCAL: 250.0}


def test_usda_gap_raises_no_coverage_warning_but_off_gap_does() -> None:
    usda = to_food_item(USDAItem(fdc_id=1, description="Curated", energy_kcal_in_100g=100.0))
    off = to_food_item(OFFItem(code="1", product_name="Crowd-sourced", energy_kcal_in_100g=100.0))

    def _warnings(food: FoodItem) -> list[str]:
        plan = MealPlan(
            user_id="x",
            meals=[Meal(kind="lunch", name="r", portions=[Portion(food=food, grams=100)], recipe=LONG_INSTRUCTIONS)],
        )
        return total_meal_plan(plan).warnings

    # A USDA food's unlisted-but-modelled nutrients (e.g. fibre) are real zeros, so not flagged.
    assert not any(w.startswith("fiber:") for w in _warnings(usda))
    # The same gap on an Open Food Facts food is a genuine unknown and is flagged.
    assert any(w.startswith("fiber:") for w in _warnings(off))
    # Salt is the one nutrient USDA does not model at all, so it stays unknown and is flagged.
    assert any(w.startswith("salt:") for w in _warnings(usda))


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
