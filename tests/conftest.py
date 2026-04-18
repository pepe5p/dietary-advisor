"""Test fixtures: redirect all on-disk state to a tmp_path so tests are hermetic."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dietary_advisor.config import get_api_keys, get_settings
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName
from dietary_advisor.schemas.profile import (
    Allergen,
    Condition,
    DietPattern,
    Sex,
    UserProfile,
)


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every Settings-derived path under `tmp_path` and reset the cache."""
    monkeypatch.setenv("DA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DA_CHROMA_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("DA_PROFILE_DB", str(tmp_path / "data" / "profiles.sqlite"))
    monkeypatch.setenv("DA_USDA_CACHE", str(tmp_path / "data" / "usda_cache.sqlite"))
    monkeypatch.setenv("DA_CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("USDA_API_KEY", os.environ.get("USDA_API_KEY", "DEMO_KEY"))
    # Provide a dummy key so pydantic-ai's openai provider can be instantiated
    # in tests; we still override the model itself with TestModel before any
    # network call is attempted.
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "test-dummy"))
    get_settings.cache_clear()
    get_api_keys.cache_clear()


@pytest.fixture
def healthy_profile() -> UserProfile:
    return UserProfile(
        user_id="t_h",
        age=30,
        sex=Sex.MALE,
        height_cm=180,
        weight_kg=78,
        activity_factor=1.55,
    )


@pytest.fixture
def vegan_peanut_profile() -> UserProfile:
    return UserProfile(
        user_id="t_v",
        age=27,
        sex=Sex.FEMALE,
        height_cm=168,
        weight_kg=62,
        allergens=[Allergen.PEANUTS],
        diet_pattern=DietPattern.VEGAN,
    )


@pytest.fixture
def hypertensive_profile() -> UserProfile:
    return UserProfile(
        user_id="t_hyp",
        age=58,
        sex=Sex.MALE,
        height_cm=174,
        weight_kg=95,
        conditions=[Condition.HYPERTENSION],
    )


@pytest.fixture
def chicken_food() -> FoodItem:
    return FoodItem(
        name="Chicken breast",
        nutrients_per_100g={
            NutrientName.ENERGY_KCAL: 165.0,
            NutrientName.PROTEIN_G: 31.0,
            NutrientName.CARBS_G: 0.0,
            NutrientName.FAT_G: 3.6,
        },
        tags=["pescatarian"],
    )


@pytest.fixture
def rice_food() -> FoodItem:
    return FoodItem(
        name="White rice cooked",
        nutrients_per_100g={
            NutrientName.ENERGY_KCAL: 130.0,
            NutrientName.PROTEIN_G: 2.7,
            NutrientName.CARBS_G: 28.0,
            NutrientName.FAT_G: 0.3,
            NutrientName.FIBER_G: 0.4,
            NutrientName.SODIUM_MG: 1.0,
        },
        tags=["vegan", "vegetarian", "pescatarian"],
    )


@pytest.fixture
def peanut_food() -> FoodItem:
    return FoodItem(
        name="Peanut butter",
        nutrients_per_100g={
            NutrientName.ENERGY_KCAL: 588.0,
            NutrientName.PROTEIN_G: 25.0,
            NutrientName.CARBS_G: 20.0,
            NutrientName.FAT_G: 50.0,
        },
        tags=["vegan", "vegetarian", "contains:peanuts"],
    )


@pytest.fixture
def salty_food() -> FoodItem:
    return FoodItem(
        name="Salted ham",
        nutrients_per_100g={
            NutrientName.ENERGY_KCAL: 145.0,
            NutrientName.PROTEIN_G: 21.0,
            NutrientName.SODIUM_MG: 1500.0,
        },
        tags=["omnivore"],
    )
