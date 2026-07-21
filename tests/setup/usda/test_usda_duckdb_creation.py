"""Setup-time document construction and schema helpers for the USDA build."""

from __future__ import annotations

from dietary_advisor.food_db.nutrients import COLUMN_TO_NUTRIENT
from dietary_advisor.totaller.nutrition import NutrientName
from setup.units import stored_column, TARGET_UNIT
from setup.usda.duckdb_creation import _build_document, _expected_foods_columns, _NUTRIENT_NUMBERS


def test_document_prose() -> None:
    doc = _build_document(
        {
            "description": "Cheese, cheddar",
            "category": "Dairy and Egg Products",
        }
    )
    assert doc == "Cheese, cheddar. A Dairy and Egg Products product."
    assert doc.strip()


def test_document_uses_an_for_vowel_category() -> None:
    doc = _build_document(
        {
            "description": "Bison, ground",
            "category": "American Indian/Alaska Native Foods",
        }
    )
    assert doc == "Bison, ground. An American Indian/Alaska Native Foods product."


def test_expected_columns_include_embedding_without_unit_columns() -> None:
    cols = _expected_foods_columns()
    assert "embedding" in cols
    assert "scientific_name" not in cols
    assert "data_type" not in cols
    for prefix in _NUTRIENT_NUMBERS:
        assert stored_column(prefix) in cols
        assert f"{prefix}_unit" not in cols
        assert TARGET_UNIT[prefix]


def test_reader_and_builder_agree_on_nutrient_prefixes() -> None:
    for prefix in _NUTRIENT_NUMBERS:
        assert stored_column(prefix) in COLUMN_TO_NUTRIENT


def test_energy_prefers_kcal_with_atwater_fallback() -> None:
    # 208 kcal is primary; the Atwater factors (957/958) are only fallbacks.
    assert _NUTRIENT_NUMBERS["energy_kcal"] == (208.0, 957.0, 958.0)
    assert COLUMN_TO_NUTRIENT["energy_kcal_in_100g"] is NutrientName.ENERGY_KCAL
