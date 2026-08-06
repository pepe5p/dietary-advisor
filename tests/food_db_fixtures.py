"""Shared seed data and builders for the tiny in-process OFF/USDA DuckDBs.

Used by the session fixture in ``tests/conftest.py`` and by the reader unit
tests. Embeddings are 3-dimensional (see ``TEST_EMBEDDING_DIM``) so semantic
ranking stays deterministic without downloading the real ONNX model; the
query embedder is stubbed via ``stub_embedders``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import duckdb
import pytest

from dietary_advisor.food_db import off_food_db as off_mod
from dietary_advisor.food_db import usda_food_db as usda_mod
from dietary_advisor.food_db.nutrients import COLUMN_TO_NUTRIENT
from dietary_advisor.food_db.off_food_db import to_code as to_off_code

# Toy embedding width. Must match the FLOAT[n] column created below and the
# DA_OFF_EMBEDDING_DIM override in the session fixture — both readers cast the
# query vector to FLOAT[{settings.off_embedding_dim}].
TEST_EMBEDDING_DIM = 3

# Prefixed code of a seeded OFF product with positive macros; evaluation tests
# that hydrate / total a portion use this instead of probing the DB at runtime.
ANY_OFF_CODE = to_off_code("111")


class OffSeed(NamedTuple):
    code: str
    product_name: str
    product_name_pl: str
    categories: str
    brands: str
    embedding: list[float]
    energy_kcal_in_100g: float
    proteins_g_in_100g: float
    carbohydrates_g_in_100g: float
    fat_g_in_100g: float
    fiber_g_in_100g: float


class UsdaSeed(NamedTuple):
    fdc_id: int
    description: str
    category: str
    embedding: list[float]
    energy_kcal_in_100g: float


# Mozzarella-like products point along axis 0, yogurt along axis 1. The stubbed
# query embedder maps queries into the same space so semantic ranking is
# deterministic.
OFF_PRODUCTS: list[OffSeed] = [
    OffSeed("111", "Mozzarella", "Mozzarella", "cheese", "Galbani", [1.0, 0.0, 0.0], 280.0, 22.0, 2.0, 20.0, 0.0),
    OffSeed(
        "222",
        "Light mozzarella cheese",
        "Ser mozzarella light",
        "cheese",
        "Piatnica",
        [0.9, 0.1, 0.0],
        180.0,
        20.0,
        2.0,
        10.0,
        0.0,
    ),
    OffSeed("333", "Natural yogurt", "Jogurt naturalny", "yogurt", "Danone", [0.0, 1.0, 0.0], 60.0, 4.0, 5.0, 3.0, 0.0),
]

USDA_FOODS: list[UsdaSeed] = [
    UsdaSeed(111, "Cheddar cheese", "Dairy and Egg Products", [1.0, 0.0, 0.0], 100.0),
    UsdaSeed(222, "Cheese, mozzarella", "Dairy and Egg Products", [0.9, 0.1, 0.0], 100.0),
    UsdaSeed(333, "Yogurt, plain", "Dairy and Egg Products", [0.0, 1.0, 0.0], 100.0),
]

# Union of the OFF and USDA query vectors used by the reader unit tests.
QUERY_VECTORS: dict[str, list[float]] = {
    "mozzarella": [1.0, 0.0, 0.0],
    "cheese": [1.0, 0.0, 0.0],
    "yogurt": [0.0, 1.0, 0.0],
    "creamy italian dairy": [0.95, 0.05, 0.0],
    "creamy dairy": [0.95, 0.05, 0.0],
}

_OFF_NUTRIENT_COLS = sorted({*COLUMN_TO_NUTRIENT, "energy_kj_in_100g", "salt_g_in_100g"})


def build_off_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    nutrient_cols = ", ".join(f"{col} DOUBLE" for col in _OFF_NUTRIENT_COLS)
    con.execute(
        "CREATE TABLE products ("
        "code VARCHAR, product_name VARCHAR, product_name_pl VARCHAR, "
        "ingredients_text VARCHAR, brands VARCHAR, brands_tags VARCHAR[], "
        "quantity VARCHAR, serving_size VARCHAR, serving_quantity DOUBLE, "
        "product_quantity DOUBLE, product_quantity_unit VARCHAR, nutrition_data_per VARCHAR, "
        "categories VARCHAR, categories_tags VARCHAR[], compared_to_category VARCHAR, "
        "labels_tags VARCHAR[], allergens_tags VARCHAR[], traces_tags VARCHAR[], "
        "additives_tags VARCHAR[], nova_group DOUBLE, nutriscore_grade VARCHAR, "
        f"nutriscore_score DOUBLE, embedding FLOAT[{TEST_EMBEDDING_DIM}], "
        f"{nutrient_cols})"
    )
    for p in OFF_PRODUCTS:
        con.execute(
            "INSERT INTO products ("
            "code, product_name, product_name_pl, brands, brands_tags, "
            "categories, categories_tags, labels_tags, allergens_tags, "
            "traces_tags, additives_tags, embedding, "
            "energy_kcal_in_100g, proteins_g_in_100g, carbohydrates_g_in_100g, "
            "fat_g_in_100g, fiber_g_in_100g"
            ") VALUES (?, ?, ?, ?, [], ?, [], [], [], [], [], ?, ?, ?, ?, ?, ?)",
            [
                p.code,
                p.product_name,
                p.product_name_pl,
                p.brands,
                p.categories,
                p.embedding,
                p.energy_kcal_in_100g,
                p.proteins_g_in_100g,
                p.carbohydrates_g_in_100g,
                p.fat_g_in_100g,
                p.fiber_g_in_100g,
            ],
        )
    con.execute("PRAGMA create_fts_index('products', 'code', 'product_name', 'product_name_pl', 'brands')")
    con.close()
    return path


def build_usda_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    nutrient_cols = ", ".join(f"{col} DOUBLE" for col in COLUMN_TO_NUTRIENT)
    con.execute(
        "CREATE TABLE foods ("
        "fdc_id BIGINT, description VARCHAR, category VARCHAR, "
        f"embedding FLOAT[{TEST_EMBEDDING_DIM}], {nutrient_cols})"
    )
    for f in USDA_FOODS:
        con.execute(
            "INSERT INTO foods (fdc_id, description, category, embedding, energy_kcal_in_100g) VALUES (?, ?, ?, ?, ?)",
            [f.fdc_id, f.description, f.category, f.embedding, f.energy_kcal_in_100g],
        )
    con.execute("PRAGMA create_fts_index('foods', 'fdc_id', 'description')")
    con.close()
    return path


def stub_embedders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(off_mod, "embed_query", lambda text: QUERY_VECTORS[text])
    monkeypatch.setattr(usda_mod, "embed_query", lambda text: QUERY_VECTORS[text])
