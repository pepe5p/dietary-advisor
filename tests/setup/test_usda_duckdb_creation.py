"""Setup-time document construction and schema helpers for the USDA build."""

from __future__ import annotations

import duckdb

from dietary_advisor.food_db.usda_food_db import _NUTRIENT_FACTORS
from dietary_advisor.schemas.nutrition import NutrientName
from setup.usda_duckdb_creation import _document_sql, _expected_foods_columns, _NUTRIENT_NUMBERS


def _document_for(**columns: object) -> str:
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE foods (description VARCHAR, category VARCHAR, scientific_name VARCHAR, data_type VARCHAR)"
        )
        keys = list(columns)
        placeholders = ", ".join("?" for _ in keys)
        insert = f"INSERT INTO foods ({', '.join(keys)}) VALUES ({placeholders})"  # noqa: S608 (test-only; hardcoded kwargs)
        con.execute(insert, list(columns.values()))
        return con.execute(f"SELECT {_document_sql()} FROM foods").fetchone()[0]  # type: ignore[index]  # noqa: S608
    finally:
        con.close()


def test_document_labels_every_valuable_column_and_spells_out_type() -> None:
    doc = _document_for(
        description="Cheese, cheddar",
        category="Dairy and Egg Products",
        scientific_name="Bos taurus",
        data_type="foundation_food",
    )
    assert doc == (
        "Name: Cheese, cheddar\nCategory: Dairy and Egg Products\nScientific name: Bos taurus\nType: Foundation"
    )


def test_document_skips_missing_parts_and_maps_sr_legacy() -> None:
    doc = _document_for(description="Carrots, raw", data_type="sr_legacy_food")
    assert doc == "Name: Carrots, raw\nType: SR Legacy"


def test_expected_columns_include_embedding_and_paired_nutrients() -> None:
    cols = _expected_foods_columns()
    assert "embedding" in cols
    for prefix in _NUTRIENT_NUMBERS:
        assert f"{prefix}_100g" in cols
        assert f"{prefix}_unit" in cols


def test_reader_and_builder_agree_on_nutrient_prefixes() -> None:
    # The runtime reader reads `<prefix>_100g`; every column it expects must be
    # one the build actually materializes.
    reader_prefixes = {col.removesuffix("_100g") for col, _, _ in _NUTRIENT_FACTORS}
    assert reader_prefixes == set(_NUTRIENT_NUMBERS)


def test_energy_prefers_kcal_with_atwater_fallback() -> None:
    # 208 kcal is primary; the Atwater factors (957/958) are only fallbacks.
    assert _NUTRIENT_NUMBERS["energy_kcal"] == (208.0, 957.0, 958.0)
    assert NutrientName.ENERGY_KCAL in {n for _, n, _ in _NUTRIENT_FACTORS}
