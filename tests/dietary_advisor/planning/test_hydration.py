"""Unit tests for `to_food_item`: mapping food_db read models to the domain FoodItem."""

from __future__ import annotations

from dietary_advisor.food_db import OFFItem, USDAItem
from dietary_advisor.planning.hydration import to_food_item
from dietary_advisor.totaller.nutrition import NutrientName


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
