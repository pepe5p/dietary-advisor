"""Sanity checks on the DSL schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dietary_advisor.schemas.constraints import HardConstraint
from dietary_advisor.schemas.meal_plan import (
    Meal,
    MealKind,
    MealPlan,
    Portion,
    Recipe,
)
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName
from dietary_advisor.schemas.profile import (
    Allergen,
    Condition,
    DietPattern,
    Sex,
    UserProfile,
)


def test_user_profile_complexity_levels() -> None:
    healthy = UserProfile(user_id="a", age=30, sex=Sex.MALE, height_cm=180, weight_kg=78)
    assert healthy.complexity_level == 1

    veggie = UserProfile(
        user_id="b",
        age=30,
        sex=Sex.FEMALE,
        height_cm=170,
        weight_kg=60,
        diet_pattern=DietPattern.VEGAN,
        allergens=[Allergen.PEANUTS],
    )
    assert veggie.complexity_level == 2

    clinical = UserProfile(
        user_id="c",
        age=58,
        sex=Sex.MALE,
        height_cm=174,
        weight_kg=95,
        conditions=[Condition.TYPE_2_DIABETES, Condition.HYPERTENSION],
    )
    assert clinical.complexity_level == 3


def test_user_profile_normalises_dislikes_and_drops_none_when_clinical() -> None:
    p = UserProfile(
        user_id="d",
        age=30,
        sex=Sex.MALE,
        height_cm=180,
        weight_kg=78,
        disliked_foods=["  Liver ", "liver", "PORK"],
        conditions=[Condition.NONE, Condition.HYPERTENSION],
    )
    assert p.disliked_foods == ["liver", "pork"]
    assert Condition.NONE not in p.conditions
    assert Condition.HYPERTENSION in p.conditions


def test_user_profile_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        UserProfile(user_id="x", age=200, sex=Sex.MALE, height_cm=180, weight_kg=78)


def test_food_item_rejects_negative_nutrients() -> None:
    with pytest.raises(ValidationError):
        FoodItem(name="Bad", nutrients_per_100g={NutrientName.PROTEIN_G: -1.0})


def test_food_item_allergen_check() -> None:
    f = FoodItem(name="PB", tags=["vegan", "contains:peanuts"])
    assert f.contains_allergen("peanuts")
    assert f.contains_allergen("PEANUTS")
    assert not f.contains_allergen("milk")


def test_meal_plan_requires_meal() -> None:
    with pytest.raises(ValidationError):
        MealPlan(user_id="x", meals=[])


def test_meal_plan_json_schema_groq_compatible() -> None:
    """Groq rejects tool schemas with propertyNames + unresolved $defs/NutrientName."""
    schema = json.dumps(MealPlan.model_json_schema())
    assert "propertyNames" not in schema
    assert "#/$defs/NutrientName" not in schema


def test_hard_constraint_factories() -> None:
    c = HardConstraint.allergen("peanuts")
    assert c.kind == "allergen_exclusion"
    assert c.target == "peanuts"

    c2 = HardConstraint.max_nutrient(NutrientName.SODIUM_MG, value=2000)
    assert c2.kind == "max_nutrient"
    assert c2.value == 2000


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
                ),
            ),
        ],
    )
    assert plan.meals[0].recipe.portions[0].grams == 150
