"""Hybrid (BM25 + semantic) food search over a synthetic USDA DuckDB.

Builds a tiny 3-dimensional-embedding DB in a tmp file and stubs the query
embedder, so fusion, degradation and fallback behaviour are exercised without
downloading the real ONNX model (mirrors tests/dietary_advisor/test_off_food_db).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from dietary_advisor.config import FoodDbUsage, get_settings
from dietary_advisor.food_db import usda_food_db as usda_mod
from dietary_advisor.food_db.usda_food_db import UsdaFoodDb
from dietary_advisor.schemas.nutrition import NutrientName

# fdc_id, description, category, data_type, embedding vector.
_FOODS = [
    (111, "Cheddar cheese", "Dairy and Egg Products", "foundation_food", [1.0, 0.0, 0.0]),
    (222, "Cheese, mozzarella", "Dairy and Egg Products", "sr_legacy_food", [0.9, 0.1, 0.0]),
    (333, "Yogurt, plain", "Dairy and Egg Products", "sr_legacy_food", [0.0, 1.0, 0.0]),
]

_QUERY_VECTORS = {
    "cheese": [1.0, 0.0, 0.0],
    "yogurt": [0.0, 1.0, 0.0],
    "creamy dairy": [0.95, 0.05, 0.0],
}


@pytest.fixture()
def usda_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DA_OFF_EMBEDDING_DIM", "3")
    get_settings.cache_clear()

    db_path = tmp_path / "usda_test.duckdb"
    con = duckdb.connect(str(db_path))
    nutrient_cols = ", ".join(f"{col} DOUBLE" for col, _, _ in usda_mod._NUTRIENT_FACTORS)
    con.execute(
        "CREATE TABLE foods ("
        "fdc_id BIGINT, description VARCHAR, category VARCHAR, data_type VARCHAR, "
        f"scientific_name VARCHAR, embedding FLOAT[3], {nutrient_cols})"
    )
    for fdc_id, description, category, data_type, vec in _FOODS:
        con.execute(
            "INSERT INTO foods (fdc_id, description, category, data_type, scientific_name, "
            "embedding, energy_kcal_100g) VALUES (?, ?, ?, ?, NULL, ?, ?)",
            [fdc_id, description, category, data_type, vec, 100.0],
        )
    con.execute("PRAGMA create_fts_index('foods', 'fdc_id', 'description')")
    con.close()
    return db_path


def _stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usda_mod, "embed_query", lambda text: _QUERY_VECTORS[text])


def test_hybrid_ranks_exact_and_semantic_match_first(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with UsdaFoodDb(usda_db_path) as db:
        results = db.search("cheese", limit=3)
    codes = [r.code for r in results]
    # Both cheeses (lexical + semantic agreement) outrank the yogurt.
    assert codes[:2] == ["usda:111", "usda:222"]
    assert "usda:333" in codes


def test_codes_are_usda_prefixed_and_nutrients_mapped(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with UsdaFoodDb(usda_db_path) as db:
        [top] = db.search("yogurt", limit=1)
    assert top.code == "usda:333"
    assert top.nutrients_per_100g[NutrientName.ENERGY_KCAL] == 100.0


def test_semantic_finds_foods_without_lexical_overlap(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with UsdaFoodDb(usda_db_path) as db:
        results = db.search("creamy dairy", limit=2)
    # No shared tokens with any indexed description, yet the embedding pulls cheese.
    assert [r.code for r in results] == ["usda:111", "usda:222"]


def test_semantic_failure_degrades_to_lexical(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_text: str) -> list[float]:
        raise RuntimeError("embedder unavailable")

    monkeypatch.setattr(usda_mod, "embed_query", _boom)
    with UsdaFoodDb(usda_db_path) as db:
        results = db.search("cheese", limit=3)
    assert {r.code for r in results} == {"usda:111", "usda:222"}


def test_only_bm25_ignores_semantic_only_query(usda_db_path: Path) -> None:
    with UsdaFoodDb(usda_db_path, usage=FoodDbUsage.ONLY_BM25) as db:
        # No shared FTS token, and the semantic channel is off, so nothing matches.
        assert db.search("creamy dairy", limit=3) == []
        assert {r.code for r in db.search("cheese", limit=3)} == {"usda:111", "usda:222"}


def test_only_semantic_uses_embedding_ranking(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with UsdaFoodDb(usda_db_path, usage=FoodDbUsage.ONLY_SEMANTIC) as db:
        results = db.search("creamy dairy", limit=2)
    assert [r.code for r in results] == ["usda:111", "usda:222"]


def test_disabled_usage_returns_nothing(usda_db_path: Path) -> None:
    with UsdaFoodDb(usda_db_path, usage=FoodDbUsage.DISABLED) as db:
        assert db.search("cheese", limit=3) == []


def test_empty_query_returns_nothing(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with UsdaFoodDb(usda_db_path) as db:
        assert db.search("   ") == []


def test_get_food_by_usda_code(usda_db_path: Path) -> None:
    with UsdaFoodDb(usda_db_path) as db:
        food = db.get_food("usda:222")
    assert food.code == "usda:222"
    assert food.name == "Cheese, mozzarella"


def test_get_food_unknown_code_raises(usda_db_path: Path) -> None:
    with UsdaFoodDb(usda_db_path) as db, pytest.raises(KeyError):
        db.get_food("usda:999999")
