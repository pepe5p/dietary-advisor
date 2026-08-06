"""Test fixtures: redirect all on-disk state to a tmp_path so tests are hermetic."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from dietary_advisor.config import get_api_keys, get_settings
from dietary_advisor.profile import UserProfile
from dietary_advisor.totaller.nutrition import FoodItem, MacroTargets, NutrientName
from evaluation.settings import get_evaluation_settings
from setup.settings import get_setup_settings
from tests.food_db_fixtures import build_off_db, build_usda_db, TEST_EMBEDDING_DIM

# Realistic placeholder meal `recipe` text for tests that don't care about the
# specific recipe content.
LONG_INSTRUCTIONS = (
    "1. Prep all ingredients: wash, chop and measure them out. "
    "2. Cook each component using the appropriate method and time. "
    "3. Combine and plate before serving."
)
LONG_RATIONALE = (
    "This day balances lean protein across meals to hit the macro targets while keeping saturated "
    "fat moderate. Wholegrains and vegetables at lunch and dinner support fibre and potassium. "
    "No supplementation is needed for this profile beyond what food provides."
)


@pytest.fixture(scope="session", autouse=True)
def _test_food_dbs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Build the tiny OFF/USDA DuckDBs once and point Settings at them for the session.

    Overrides the unreachable sentinel paths from ``[tool.pytest_env]``. The
    embedding dim must match the FLOAT[n] columns the builders create.
    """
    db_dir = tmp_path_factory.mktemp("food_db")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DA_OFF_EMBEDDING_DIM", str(TEST_EMBEDDING_DIM))
        mp.setenv("DA_OFF_DB", str(build_off_db(db_dir / "off.duckdb")))
        mp.setenv("DA_USDA_DB", str(build_usda_db(db_dir / "usda.duckdb")))
        get_settings.cache_clear()
        yield


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every Settings-derived path under `tmp_path` and reset the cache."""
    monkeypatch.setenv("DA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DA_CHROMA_DIR", str(tmp_path / "data" / "chroma"))
    # Provide a dummy key so pydantic-ai's openai provider can be instantiated
    # in tests; we still override the model itself with TestModel before any
    # network call is attempted.
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "test-dummy"))
    get_settings.cache_clear()
    get_api_keys.cache_clear()
    get_setup_settings.cache_clear()
    get_evaluation_settings.cache_clear()


@pytest.fixture()
def healthy_profile() -> UserProfile:
    return UserProfile(
        user_id="t_h",
        age=30,
        sex="male",
        height_cm=180,
        weight_kg=78,
        targets=MacroTargets(energy_kcal=2500.0, protein_g=125.0, carbs_g=310.0, fat_g=70.0),
    )


@pytest.fixture()
def vegan_peanut_profile() -> UserProfile:
    return UserProfile(
        user_id="t_v",
        age=27,
        sex="female",
        height_cm=168,
        weight_kg=62,
        allergens=["peanuts"],
        diet_pattern="vegan",
        targets=MacroTargets(energy_kcal=2000.0, protein_g=90.0, carbs_g=250.0, fat_g=55.0),
    )


@pytest.fixture()
def hypertensive_profile() -> UserProfile:
    return UserProfile(
        user_id="t_hyp",
        age=58,
        sex="male",
        height_cm=174,
        weight_kg=95,
        conditions=["hypertension"],
        targets=MacroTargets(energy_kcal=2200.0, protein_g=140.0, carbs_g=220.0, fat_g=65.0),
    )


@pytest.fixture()
def chicken_food() -> FoodItem:
    return FoodItem(
        name="Chicken breast",
        nutrients_per_100g={
            NutrientName.ENERGY_KCAL: 165.0,
            NutrientName.PROTEIN_G: 31.0,
            NutrientName.CARBS_G: 0.0,
            NutrientName.FAT_G: 3.6,
        },
    )


@pytest.fixture()
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
    )


@pytest.fixture()
def peanut_food() -> FoodItem:
    return FoodItem(
        name="Peanut butter",
        nutrients_per_100g={
            NutrientName.ENERGY_KCAL: 588.0,
            NutrientName.PROTEIN_G: 25.0,
            NutrientName.CARBS_G: 20.0,
            NutrientName.FAT_G: 50.0,
        },
    )


@pytest.fixture()
def salty_food() -> FoodItem:
    return FoodItem(
        name="Salted ham",
        nutrients_per_100g={
            NutrientName.ENERGY_KCAL: 145.0,
            NutrientName.PROTEIN_G: 21.0,
            NutrientName.SODIUM_MG: 1500.0,
        },
    )
