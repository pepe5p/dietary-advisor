"""Hybrid (BM25 + semantic) product search over the session-scoped OFF DuckDB.

The DB itself is built once in ``tests.conftest``; this module stubs the query
embedder so fusion, degradation and fallback behaviour are exercised without
downloading the real ONNX model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dietary_advisor.config import FoodDbUsage, get_settings
from dietary_advisor.food_db import off_food_db as off_mod
from dietary_advisor.food_db.errors import OFFUnknownFoodCodeError
from dietary_advisor.food_db.fusion import reciprocal_rank_fusion
from dietary_advisor.food_db.off_food_db import OffFoodDb, to_code
from tests.food_db_fixtures import stub_embedders


@pytest.fixture()
def off_db_path() -> Path:
    return get_settings().off_db


def test_hybrid_ranks_exact_and_semantic_match_first(off_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedders(monkeypatch)
    with OffFoodDb(off_db_path) as db:
        results = db.search("mozzarella", limit=3)
    codes = [r.code for r in results]
    # Both mozzarella products (lexical + semantic agreement) outrank the yogurt.
    assert codes[:2] == ["111", "222"]
    assert "333" in codes


def test_semantic_finds_products_without_lexical_overlap(off_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedders(monkeypatch)
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
    stub_embedders(monkeypatch)
    with OffFoodDb(off_db_path, usage=FoodDbUsage.ONLY_SEMANTIC) as db:
        results = db.search("creamy italian dairy", limit=2)
    assert [r.code for r in results] == ["111", "222"]


def test_disabled_usage_returns_nothing(off_db_path: Path) -> None:
    with OffFoodDb(off_db_path, usage=FoodDbUsage.DISABLED) as db:
        assert db.search("mozzarella", limit=3) == []


def test_empty_query_returns_nothing(off_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedders(monkeypatch)
    with OffFoodDb(off_db_path) as db:
        assert db.search("   ") == []


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    list_a = [{"code": "x"}, {"code": "y"}, {"code": "z"}]
    list_b = [{"code": "y"}, {"code": "w"}]
    fused = reciprocal_rank_fusion([list_a, list_b], key=lambda r: r["code"])
    # "y" is high in both lists, so it wins despite not topping either alone.
    assert fused[0]["code"] == "y"
    assert {r["code"] for r in fused} == {"x", "y", "z", "w"}


def test_get_food_accepts_prefixed_and_bare_code(off_db_path: Path) -> None:
    with OffFoodDb(off_db_path) as db:
        prefixed = db.get_food("off:222")
        bare = db.get_food("222")
    assert prefixed.code == "222"
    assert bare.code == "222"
    assert to_code(prefixed.code) == "off:222"


def test_get_food_unknown_code_raises(off_db_path: Path) -> None:
    with OffFoodDb(off_db_path) as db, pytest.raises(OFFUnknownFoodCodeError):
        db.get_food("off:999999")
