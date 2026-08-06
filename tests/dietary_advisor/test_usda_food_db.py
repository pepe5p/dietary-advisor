"""Hybrid (BM25 + semantic) food search over the session-scoped USDA DuckDB.

Mirrors ``test_off_food_db``: the DB is built once in ``tests.conftest`` and
the query embedder is stubbed so fusion / degradation stay deterministic
without the real ONNX model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dietary_advisor.config import FoodDbUsage, get_settings
from dietary_advisor.food_db import usda_food_db as usda_mod
from dietary_advisor.food_db.errors import USDAUnknownFoodCodeError
from dietary_advisor.food_db.usda_food_db import to_code, UsdaFoodDb
from tests.food_db_fixtures import stub_embedders


@pytest.fixture()
def usda_db_path() -> Path:
    return get_settings().usda_db


def test_hybrid_ranks_exact_and_semantic_match_first(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedders(monkeypatch)
    with UsdaFoodDb(usda_db_path) as db:
        results = db.search("cheese", limit=3)
    codes = [to_code(r.fdc_id) for r in results]
    # Both cheeses (lexical + semantic agreement) outrank the yogurt.
    assert codes[:2] == ["usda:111", "usda:222"]
    assert "usda:333" in codes


def test_codes_are_usda_prefixed_and_nutrients_mapped(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedders(monkeypatch)
    with UsdaFoodDb(usda_db_path) as db:
        [top] = db.search("yogurt", limit=1)
    assert to_code(top.fdc_id) == "usda:333"
    assert top.energy_kcal_in_100g == 100.0


def test_semantic_finds_foods_without_lexical_overlap(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedders(monkeypatch)
    with UsdaFoodDb(usda_db_path) as db:
        results = db.search("creamy dairy", limit=2)
    # No shared tokens with any indexed description, yet the embedding pulls cheese.
    assert [to_code(r.fdc_id) for r in results] == ["usda:111", "usda:222"]


def test_semantic_failure_degrades_to_lexical(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_text: str) -> list[float]:
        raise RuntimeError("embedder unavailable")

    monkeypatch.setattr(usda_mod, "embed_query", _boom)
    with UsdaFoodDb(usda_db_path) as db:
        results = db.search("cheese", limit=3)
    assert {to_code(r.fdc_id) for r in results} == {"usda:111", "usda:222"}


def test_only_bm25_ignores_semantic_only_query(usda_db_path: Path) -> None:
    with UsdaFoodDb(usda_db_path, usage=FoodDbUsage.ONLY_BM25) as db:
        # No shared FTS token, and the semantic channel is off, so nothing matches.
        assert db.search("creamy dairy", limit=3) == []
        assert {to_code(r.fdc_id) for r in db.search("cheese", limit=3)} == {"usda:111", "usda:222"}


def test_only_semantic_uses_embedding_ranking(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedders(monkeypatch)
    with UsdaFoodDb(usda_db_path, usage=FoodDbUsage.ONLY_SEMANTIC) as db:
        results = db.search("creamy dairy", limit=2)
    assert [to_code(r.fdc_id) for r in results] == ["usda:111", "usda:222"]


def test_disabled_usage_returns_nothing(usda_db_path: Path) -> None:
    with UsdaFoodDb(usda_db_path, usage=FoodDbUsage.DISABLED) as db:
        assert db.search("cheese", limit=3) == []


def test_empty_query_returns_nothing(usda_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedders(monkeypatch)
    with UsdaFoodDb(usda_db_path) as db:
        assert db.search("   ") == []


def test_get_food_by_usda_code(usda_db_path: Path) -> None:
    with UsdaFoodDb(usda_db_path) as db:
        food = db.get_food("usda:222")
    assert food.fdc_id == 222
    assert food.description == "Cheese, mozzarella"


def test_get_food_unknown_code_raises(usda_db_path: Path) -> None:
    with UsdaFoodDb(usda_db_path) as db, pytest.raises(USDAUnknownFoodCodeError):
        db.get_food("usda:999999")
