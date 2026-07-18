"""The FoodDb facade: code-prefix routing and grouped dual-source search."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from dietary_advisor.config import get_settings
from dietary_advisor.food_db import FoodDb, OffFoodDb, OFFItem, UsdaFoodDb, USDAItem
from dietary_advisor.food_db.facade import LookupQuery
from dietary_advisor.schemas.nutrition import NutrientName


class _FakeDb:
    """A stand-in food source that both searches and resolves codes."""

    def __init__(self, label: str, items: list[OFFItem | USDAItem] | None = None, *, boom: bool = False) -> None:
        self.label = label
        self._items = items or []
        self._boom = boom

    def search(self, query: str, limit: int = 5) -> list[OFFItem | USDAItem]:  # noqa: ARG002
        if self._boom:
            raise RuntimeError("db exploded")
        return self._items[:limit]

    def get_food(self, code: str) -> OFFItem:
        return OFFItem(code=code, name=f"{self.label}:{code}")


def _facade(off: _FakeDb | None = None, usda: _FakeDb | None = None) -> FoodDb:
    """Build a `FoodDb` around duck-typed fakes standing in for the concrete readers."""
    return FoodDb(cast("OffFoodDb | None", off), cast("UsdaFoodDb | None", usda))


def _off_item(code: str) -> OFFItem:
    return OFFItem(code=code, name=f"OFF {code}", brands="Acme", nutrients_per_100g={NutrientName.ENERGY_KCAL: 200.0})


def _usda_item(code: str) -> USDAItem:
    return USDAItem(code=code, name=f"USDA {code}", category="Dairy")


# --- get_food routing -------------------------------------------------------


def test_usda_prefix_routes_to_usda_db() -> None:
    router = _facade(_FakeDb("off"), _FakeDb("usda"))
    assert router.get_food("usda:123").name == "usda:usda:123"


def test_plain_barcode_routes_to_off_db() -> None:
    router = _facade(_FakeDb("off"), _FakeDb("usda"))
    assert router.get_food("5901234").name == "off:5901234"


def test_usda_code_without_usda_db_raises() -> None:
    router = _facade(_FakeDb("off"), None)
    with pytest.raises(KeyError):
        router.get_food("usda:123")
    assert router.get_food("5901234").name == "off:5901234"


def test_off_code_without_off_db_raises() -> None:
    router = _facade(None, _FakeDb("usda"))
    with pytest.raises(KeyError):
        router.get_food("5901234")
    assert router.get_food("usda:123").name == "usda:usda:123"


# --- grouped lookup ---------------------------------------------------------


def test_returns_both_sources_under_distinct_keys() -> None:
    off = _FakeDb("off", [_off_item("1"), _off_item("2")])
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(_facade(off, usda).lookup([LookupQuery("milk", off_limit=2, usda_limit=2)]))

    assert set(result) == {"open_food_facts", "usda"}
    assert [h["code"] for h in result["open_food_facts"]] == ["1", "2"]
    assert [h["code"] for h in result["usda"]] == ["usda:10"]
    # Source-specific context is surfaced per channel.
    assert result["open_food_facts"][0]["brands"] == "Acme"
    assert result["usda"][0]["category"] == "Dairy"


def test_missing_usda_source_yields_empty_usda_channel() -> None:
    result = asyncio.run(
        _facade(_FakeDb("off", [_off_item("1")]), None).lookup([LookupQuery("milk", off_limit=5, usda_limit=5)])
    )
    assert [h["code"] for h in result["open_food_facts"]] == ["1"]
    assert result["usda"] == []


def test_missing_off_source_yields_empty_off_channel() -> None:
    result = asyncio.run(
        _facade(None, _FakeDb("usda", [_usda_item("usda:10")])).lookup([LookupQuery("milk", off_limit=5, usda_limit=5)])
    )
    assert result["open_food_facts"] == []
    assert [h["code"] for h in result["usda"]] == ["usda:10"]


def test_one_failing_source_does_not_sink_the_other() -> None:
    off = _FakeDb("off", boom=True)
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(_facade(off, usda).lookup([LookupQuery("milk", off_limit=5, usda_limit=5)]))
    assert result["open_food_facts"] == []
    assert [h["code"] for h in result["usda"]] == ["usda:10"]


def test_batch_pools_hits_across_queries() -> None:
    off = _FakeDb("off", [_off_item("1")])
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(
        _facade(off, usda).lookup(
            [
                LookupQuery("milk", off_limit=1, usda_limit=1),
                LookupQuery("cheese", off_limit=1, usda_limit=1),
                LookupQuery("yogurt", off_limit=1, usda_limit=1),
            ]
        )
    )
    # One hit per query per source.
    assert len(result["open_food_facts"]) == 3
    assert len(result["usda"]) == 3


def test_each_query_uses_its_own_limit() -> None:
    off = _FakeDb("off", [_off_item(str(i)) for i in range(5)])
    usda = _FakeDb("usda", [_usda_item(f"usda:{i}") for i in range(5)])
    result = asyncio.run(
        _facade(off, usda).lookup(
            [
                LookupQuery("chicken", off_limit=2, usda_limit=2),
                LookupQuery("lays", off_limit=5, usda_limit=5),
            ]
        )
    )
    # 2 hits for the first query + 5 for the second, per source.
    assert len(result["open_food_facts"]) == 7
    assert len(result["usda"]) == 7


def test_per_source_limit_of_zero_skips_that_source() -> None:
    off = _FakeDb("off", [_off_item(str(i)) for i in range(5)])
    usda = _FakeDb("usda", [_usda_item(f"usda:{i}") for i in range(5)])

    staple = asyncio.run(_facade(off, usda).lookup([LookupQuery("carrot", off_limit=0, usda_limit=3)]))
    assert staple["open_food_facts"] == []
    assert len(staple["usda"]) == 3

    branded = asyncio.run(_facade(off, usda).lookup([LookupQuery("lays classic", off_limit=5, usda_limit=0)]))
    assert len(branded["open_food_facts"]) == 5
    assert branded["usda"] == []


# --- open() honours usage settings -----------------------------------------


def test_open_skips_sources_disabled_by_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DA_OFF_USAGE", "disabled")
    monkeypatch.setenv("DA_USDA_USAGE", "disabled")
    get_settings.cache_clear()
    with FoodDb.open() as db:
        assert asyncio.run(db.lookup([LookupQuery("milk", off_limit=3, usda_limit=3)])) == {
            "open_food_facts": [],
            "usda": [],
        }
        with pytest.raises(KeyError):
            db.get_food("5901234")


def test_open_raises_loudly_when_enabled_db_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DA_OFF_USAGE", "full")
    monkeypatch.setenv("DA_USDA_USAGE", "disabled")
    monkeypatch.setenv("DA_OFF_DB", str(tmp_path / "missing.duckdb"))
    get_settings.cache_clear()
    with pytest.raises(FileNotFoundError):
        FoodDb.open()
