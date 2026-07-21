"""Hybrid (BM25 + semantic) product search over a synthetic OFF DuckDB.

Builds a tiny 3-dimensional-embedding DB in a tmp file and stubs the query
embedder, so the fusion, degradation and fallback behaviour are exercised
without downloading the real ONNX model.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from dietary_advisor.config import FoodDbUsage, get_settings
from dietary_advisor.food_db import off_food_db as off_mod
from dietary_advisor.food_db.nutrients import COLUMN_TO_NUTRIENT
from dietary_advisor.food_db.off_food_db import _reciprocal_rank_fusion, OffFoodDb

# Toy 3-d embedding space: mozzarella-like products point along axis 0, yogurt
# along axis 1. The stubbed query embedder (below) maps queries into the same
# space so semantic ranking is deterministic.
_PRODUCTS = [
    ("111", "Mozzarella", "Mozzarella", "cheese", "Galbani", [1.0, 0.0, 0.0]),
    ("222", "Light mozzarella cheese", "Ser mozzarella light", "cheese", "Piatnica", [0.9, 0.1, 0.0]),
    ("333", "Natural yogurt", "Jogurt naturalny", "yogurt", "Danone", [0.0, 1.0, 0.0]),
]
# (code, product_name, product_name_pl, categories, brands, embedding)

_QUERY_VECTORS = {
    "mozzarella": [1.0, 0.0, 0.0],
    "yogurt": [0.0, 1.0, 0.0],
    "creamy italian dairy": [0.95, 0.05, 0.0],
}

_NUTRIENT_COLS = sorted({*COLUMN_TO_NUTRIENT, "energy_kj_in_100g", "salt_g_in_100g"})


@pytest.fixture()
def off_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DA_OFF_EMBEDDING_DIM", "3")
    get_settings.cache_clear()

    db_path = tmp_path / "off_test.duckdb"
    con = duckdb.connect(str(db_path))
    nutrient_cols = ", ".join(f"{col} DOUBLE" for col in _NUTRIENT_COLS)
    con.execute(
        "CREATE TABLE products ("
        "code VARCHAR, product_name VARCHAR, product_name_pl VARCHAR, "
        "ingredients_text VARCHAR, brands VARCHAR, brands_tags VARCHAR[], "
        "quantity VARCHAR, serving_size VARCHAR, serving_quantity DOUBLE, "
        "product_quantity DOUBLE, product_quantity_unit VARCHAR, nutrition_data_per VARCHAR, "
        "categories VARCHAR, categories_tags VARCHAR[], compared_to_category VARCHAR, "
        "labels_tags VARCHAR[], allergens_tags VARCHAR[], traces_tags VARCHAR[], "
        "additives_tags VARCHAR[], nova_group DOUBLE, nutriscore_grade VARCHAR, "
        "nutriscore_score DOUBLE, embedding FLOAT[3], "
        f"{nutrient_cols})"
    )
    for code, name, name_pl, categories, brand, vec in _PRODUCTS:
        con.execute(
            "INSERT INTO products (code, product_name, product_name_pl, "
            "brands, brands_tags, categories, categories_tags, "
            "labels_tags, allergens_tags, traces_tags, additives_tags, embedding) "
            "VALUES (?, ?, ?, ?, [], ?, [], [], [], [], [], ?)",
            [code, name, name_pl, brand, categories, vec],
        )
    con.execute("PRAGMA create_fts_index('products', 'code', 'product_name', 'product_name_pl', 'brands')")
    con.close()
    return db_path


def _stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(off_mod, "embed_query", lambda text: _QUERY_VECTORS[text])


def test_hybrid_ranks_exact_and_semantic_match_first(off_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with OffFoodDb(off_db_path) as db:
        results = db.search("mozzarella", limit=3)
    codes = [r.code for r in results]
    # Both mozzarella products (lexical + semantic agreement) outrank the yogurt.
    assert codes[:2] == ["111", "222"]
    assert "333" in codes


def test_semantic_finds_products_without_lexical_overlap(off_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with OffFoodDb(off_db_path) as db:
        results = db.search("creamy italian dairy", limit=2)
    # No shared tokens with any indexed name/brand, yet the embedding pulls mozzarella.
    assert [r.code for r in results] == ["111", "222"]


def test_semantic_failure_degrades_to_lexical(off_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_text: str) -> list[float]:
        raise RuntimeError("embedder unavailable")

    monkeypatch.setattr(off_mod, "embed_query", _boom)
    with OffFoodDb(off_db_path) as db:
        results = db.search("mozzarella", limit=3)
    codes = {r.code for r in results}
    assert codes == {"111", "222"}


def test_only_bm25_ignores_semantic_only_query(off_db_path: Path) -> None:
    with OffFoodDb(off_db_path, usage=FoodDbUsage.ONLY_BM25) as db:
        # No shared FTS token, and the semantic channel is off, so nothing matches.
        assert db.search("creamy italian dairy", limit=3) == []
        assert {r.code for r in db.search("mozzarella", limit=3)} == {"111", "222"}


def test_only_semantic_uses_embedding_ranking(off_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with OffFoodDb(off_db_path, usage=FoodDbUsage.ONLY_SEMANTIC) as db:
        results = db.search("creamy italian dairy", limit=2)
    assert [r.code for r in results] == ["111", "222"]


def test_disabled_usage_returns_nothing(off_db_path: Path) -> None:
    with OffFoodDb(off_db_path, usage=FoodDbUsage.DISABLED) as db:
        assert db.search("mozzarella", limit=3) == []


def test_empty_query_returns_nothing(off_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_embedder(monkeypatch)
    with OffFoodDb(off_db_path) as db:
        assert db.search("   ") == []


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    list_a = [{"code": "x"}, {"code": "y"}, {"code": "z"}]
    list_b = [{"code": "y"}, {"code": "w"}]
    fused = _reciprocal_rank_fusion([list_a, list_b], key=lambda r: r["code"])
    # "y" is high in both lists, so it wins despite not topping either alone.
    assert fused[0]["code"] == "y"
    assert {r["code"] for r in fused} == {"x", "y", "z", "w"}
