"""Unit tests for `to_food_item`: mapping food_db read models to the domain FoodItem."""

from __future__ import annotations

from dietary_advisor.food_db import OFFItem, USDAItem
from dietary_advisor.hydration import to_food_item
from dietary_advisor.schemas.nutrition import NutrientName


def test_off_item_maps_to_food_item() -> None:
    off = OFFItem(
        code="123",
        name="Mozzarella",
        description="soft cheese",
        nutrients_per_100g={NutrientName.ENERGY_KCAL: 250.0},
        tags=["vegetarian"],
        brands="Acme",
    )
    food = to_food_item(off)
    assert food.code == "123"
    assert food.name == "Mozzarella"
    assert food.description == "soft cheese"
    assert food.nutrients_per_100g == {NutrientName.ENERGY_KCAL: 250.0}
    assert food.tags == ["vegetarian"]


def test_usda_item_maps_to_food_item() -> None:
    usda = USDAItem(
        code="usda:456",
        name="Cheddar cheese",
        description="Cheddar cheese, USDA",
        nutrients_per_100g={NutrientName.ENERGY_KCAL: 400.0},
        category="Dairy",
    )
    food = to_food_item(usda)
    assert food.code == "usda:456"
    assert food.name == "Cheddar cheese"
    assert food.nutrients_per_100g == {NutrientName.ENERGY_KCAL: 400.0}
    # USDA items carry no tags (the source has none).
    assert food.tags == []
