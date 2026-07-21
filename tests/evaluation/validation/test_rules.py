"""Tests for the code-based hard rule helpers."""

from __future__ import annotations

from dietary_advisor.totaller.nutrition import FoodItem
from evaluation.validation.rules import food_contains_allergen


def test_food_contains_allergen() -> None:
    f = FoodItem(name="PB", tags=["vegan", "contains:peanuts"])
    assert food_contains_allergen(f, "peanuts")
    assert food_contains_allergen(f, "PEANUTS")
    assert not food_contains_allergen(f, "milk")
