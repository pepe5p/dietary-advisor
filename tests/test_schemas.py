"""Sanity checks on the DSL schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dietary_advisor.schemas.meal_plan import (
    Meal,
    MealKind,
    MealPlan,
    Portion,
    Recipe,
)
from dietary_advisor.schemas.nutrition import FoodItem, MacroTargets, NutrientName
from dietary_advisor.schemas.profile import UserProfile
from tests.conftest import LONG_INSTRUCTIONS

_TARGETS = MacroTargets(energy_kcal=2200.0, protein_g=120.0, carbs_g=250.0, fat_g=65.0)


def test_user_profile_normalises_dislikes_and_drops_none_when_clinical() -> None:
    p = UserProfile(
        user_id="d",
        age=30,
        sex="male",
        height_cm=180,
        weight_kg=78,
        disliked_foods=["  Liver ", "liver", "PORK"],
        conditions=["none", "hypertension"],
        targets=_TARGETS,
    )
    assert p.disliked_foods == ["liver", "pork"]
    assert "none" not in p.conditions
    assert "hypertension" in p.conditions


def test_user_profile_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        UserProfile(user_id="x", age=200, sex="male", height_cm=180, weight_kg=78, targets=_TARGETS)


def test_food_item_rejects_negative_nutrients() -> None:
    with pytest.raises(ValidationError):
        FoodItem(name="Bad", nutrients_per_100g={NutrientName.PROTEIN_G: -1.0})


def test_food_item_off_record_requires_code() -> None:
    with pytest.raises(ValidationError):
        FoodItem(name="Bad", from_open_food_facts=True)


def test_food_item_off_record_with_code_is_valid() -> None:
    item = FoodItem(code="123", name="Rice", from_open_food_facts=True)
    assert item.code == "123"


def test_food_item_llm_generated_may_omit_code() -> None:
    item = FoodItem(name="Homemade stew", from_open_food_facts=False)
    assert item.code is None


def test_meal_plan_requires_meal() -> None:
    with pytest.raises(ValidationError):
        MealPlan(user_id="x", meals=[])


def test_meal_plan_json_schema_groq_compatible() -> None:
    """Groq rejects tool schemas with propertyNames + unresolved $defs/NutrientName."""
    schema = json.dumps(MealPlan.model_json_schema())
    assert "propertyNames" not in schema
    assert "#/$defs/NutrientName" not in schema


def test_meal_plan_construction(chicken_food: FoodItem, rice_food: FoodItem) -> None:
    plan = MealPlan(
        user_id="x",
        meals=[
            Meal(
                kind=MealKind.LUNCH,
                recipe=Recipe(
                    name="Chicken bowl",
                    portions=[
                        Portion(food=chicken_food, grams=150),
                        Portion(food=rice_food, grams=200),
                    ],
                    instructions=LONG_INSTRUCTIONS,
                ),
            ),
        ],
    )
    assert plan.meals[0].recipe.portions[0].grams == 150


def test_recipe_rejects_short_instructions(chicken_food: FoodItem) -> None:
    """A one-liner like "Cook." should not pass as a real recipe."""
    with pytest.raises(ValidationError):
        Recipe(name="r", portions=[Portion(food=chicken_food, grams=100)], instructions="Cook.")
