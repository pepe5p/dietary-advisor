"""The FoodDb facade: code-prefix routing and grouped dual-source search."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from dietary_advisor.config import get_settings
from dietary_advisor.food_db import (
    FoodDb,
    OffFoodDb,
    OFFItem,
    OFFUnknownFoodCodeError,
    UnknownFoodCodeError,
    UsdaFoodDb,
    USDAItem,
    USDAUnknownFoodCodeError,
)
from dietary_advisor.food_db.facade import BatchLookupResult, LookupQuery, LookupResult, OFFHit, QueryLookupResult
from dietary_advisor.food_db.off_food_db import get_off_item_name
from dietary_advisor.food_db.usda_food_db import get_usda_item_name


class _FakeDb:
    """A stand-in food source that both searches and resolves codes."""

    def __init__(
        self,
        label: str,
        items: list[OFFItem | USDAItem] | None = None,
        *,
        items_by_query: dict[str, list[OFFItem | USDAItem]] | None = None,
        boom: bool = False,
    ) -> None:
        self.label = label
        self._items = items or []
        self._items_by_query = items_by_query
        self._boom = boom

    def search(self, query: str, limit: int = 5) -> list[OFFItem | USDAItem]:
        if self._boom:
            raise RuntimeError("db exploded")
        if self._items_by_query is not None:
            return self._items_by_query.get(query, [])[:limit]
        return self._items[:limit]

    def get_food(self, code: str) -> OFFItem | USDAItem:
        if self.label == "usda":
            return USDAItem(fdc_id=int(code.removeprefix("usda:")), description=f"{self.label}:{code}")
        return OFFItem(code=code.removeprefix("off:"), product_name=f"{self.label}:{code}")


def _facade(off: _FakeDb | None = None, usda: _FakeDb | None = None) -> FoodDb:
    """Build a `FoodDb` around duck-typed fakes standing in for the concrete readers."""
    return FoodDb(cast("OffFoodDb | None", off), cast("UsdaFoodDb | None", usda))


def _off_item(code: str) -> OFFItem:
    return OFFItem(code=code, product_name=f"OFF {code}", brands="Acme", energy_kcal_in_100g=200.0)


def _usda_item(code: str) -> USDAItem:
    fdc_id = int(code.removeprefix("usda:"))
    return USDAItem(fdc_id=fdc_id, description=f"USDA {code}", category="Dairy")


def _item_name(item: OFFItem | USDAItem) -> str:
    return get_off_item_name(item) if isinstance(item, OFFItem) else get_usda_item_name(item)


def _off(result: BatchLookupResult, i: int = 0) -> list[str]:
    return [h.code for h in result.queries[i].results.open_food_facts]


def _usda(result: BatchLookupResult, i: int = 0) -> list[str]:
    return [h.code for h in result.queries[i].results.usda]


# --- get_food routing -------------------------------------------------------


def test_usda_prefix_routes_to_usda_db() -> None:
    router = _facade(_FakeDb("off"), _FakeDb("usda"))
    assert _item_name(router.get_food("usda:123")) == "usda:usda:123"


def test_off_prefix_routes_to_off_db() -> None:
    router = _facade(_FakeDb("off"), _FakeDb("usda"))
    assert _item_name(router.get_food("off:5901234")) == "off:off:5901234"


def test_unprefixed_code_raises() -> None:
    router = _facade(_FakeDb("off"), _FakeDb("usda"))
    with pytest.raises(UnknownFoodCodeError, match="off:.*usda:"):
        router.get_food("5901234")


def test_usda_code_without_usda_db_raises() -> None:
    router = _facade(_FakeDb("off"), None)
    with pytest.raises(USDAUnknownFoodCodeError):
        router.get_food("usda:123")
    assert _item_name(router.get_food("off:5901234")) == "off:off:5901234"


def test_off_code_without_off_db_raises() -> None:
    router = _facade(None, _FakeDb("usda"))
    with pytest.raises(OFFUnknownFoodCodeError):
        router.get_food("off:5901234")
    assert _item_name(router.get_food("usda:123")) == "usda:usda:123"


# --- grouped lookup ---------------------------------------------------------


def test_returns_both_sources_under_distinct_keys() -> None:
    off = _FakeDb("off", [_off_item("1"), _off_item("2")])
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(_facade(off, usda).lookup([LookupQuery(query="milk", max_results_off=2, max_results_usda=2)]))

    assert _off(result) == ["off:1", "off:2"]
    assert _usda(result) == ["usda:10"]
    off_hits = result.queries[0].results.open_food_facts
    assert off_hits[0].brands == "Acme"
    assert off_hits[0].energy_kcal == 200.0
    assert result.queries[0].results.usda[0].category == "Dairy"


def test_lookup_csv_renders_both_sources() -> None:
    off = _FakeDb("off", [_off_item("1")])
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(_facade(off, usda).lookup([LookupQuery(query="milk", max_results_off=1, max_results_usda=1)]))
    csv_text = result.render_csv()
    assert "# query: milk" in csv_text
    assert "## open_food_facts" in csv_text
    assert "## usda" in csv_text
    assert "energy_kcal" in csv_text
    assert "sodium_mg" in csv_text
    assert "off:1" in csv_text
    assert "usda:10" in csv_text


def test_lookup_json_renders_both_sources() -> None:
    off = _FakeDb("off", [_off_item("1")])
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(_facade(off, usda).lookup([LookupQuery(query="milk", max_results_off=1, max_results_usda=1)]))
    json_text = result.render_json()
    assert '"query":"milk"' in json_text
    assert '"open_food_facts"' in json_text
    assert '"usda"' in json_text
    assert '"energy_kcal"' in json_text
    assert '"off:1"' in json_text
    assert '"usda:10"' in json_text


def test_empty_batch_render_json() -> None:
    assert BatchLookupResult().render_json() == '{"queries":[]}'


def test_missing_usda_source_yields_empty_usda_channel() -> None:
    query = LookupQuery(query="milk", max_results_off=5, max_results_usda=5)
    result = asyncio.run(_facade(_FakeDb("off", [_off_item("1")]), None).lookup([query]))
    assert _off(result) == ["off:1"]
    assert result.queries[0].results.usda == []


def test_missing_off_source_yields_empty_off_channel() -> None:
    query = LookupQuery(query="milk", max_results_off=5, max_results_usda=5)
    result = asyncio.run(_facade(None, _FakeDb("usda", [_usda_item("usda:10")])).lookup([query]))
    assert result.queries[0].results.open_food_facts == []
    assert _usda(result) == ["usda:10"]


def test_one_failing_source_does_not_sink_the_other() -> None:
    off = _FakeDb("off", boom=True)
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(_facade(off, usda).lookup([LookupQuery(query="milk", max_results_off=5, max_results_usda=5)]))
    assert result.queries[0].results.open_food_facts == []
    assert _usda(result) == ["usda:10"]


def test_batch_returns_one_section_per_query() -> None:
    off = _FakeDb("off", [_off_item("1")])
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(
        _facade(off, usda).lookup(
            [
                LookupQuery(query="milk", max_results_off=1, max_results_usda=1),
                LookupQuery(query="cheese", max_results_off=1, max_results_usda=1),
                LookupQuery(query="yogurt", max_results_off=1, max_results_usda=1),
            ]
        )
    )
    assert len(result.queries) == 3
    assert all(_off(result, i) == ["off:1"] for i in range(3))
    assert all(_usda(result, i) == ["usda:10"] for i in range(3))


def test_each_query_uses_its_own_limit() -> None:
    off = _FakeDb("off", [_off_item(str(i)) for i in range(5)])
    usda = _FakeDb("usda", [_usda_item(f"usda:{i}") for i in range(5)])
    result = asyncio.run(
        _facade(off, usda).lookup(
            [
                LookupQuery(query="chicken", max_results_off=2, max_results_usda=2),
                LookupQuery(query="lays", max_results_off=5, max_results_usda=5),
            ]
        )
    )
    assert len(_off(result, 0)) == 2
    assert len(_usda(result, 0)) == 2
    assert len(_off(result, 1)) == 5
    assert len(_usda(result, 1)) == 5


def test_hits_are_attributed_to_the_matching_query() -> None:
    off = _FakeDb("off", items_by_query={"bread": [_off_item("1")], "milk": [_off_item("2")]})
    usda = _FakeDb("usda", items_by_query={"bread": [_usda_item("usda:10")], "milk": [_usda_item("usda:20")]})
    result = asyncio.run(
        _facade(off, usda).lookup(
            [
                LookupQuery(query="bread", max_results_off=1, max_results_usda=1),
                LookupQuery(query="milk", max_results_off=1, max_results_usda=1),
            ]
        )
    )
    assert _off(result, 0) == ["off:1"]
    assert _usda(result, 0) == ["usda:10"]
    assert _off(result, 1) == ["off:2"]
    assert _usda(result, 1) == ["usda:20"]


def test_batch_csv_renders_one_query_section_per_input() -> None:
    off = _FakeDb("off", [_off_item("1")])
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(
        _facade(off, usda).lookup(
            [
                LookupQuery(query="milk", max_results_off=1, max_results_usda=1),
                LookupQuery(query="cheese", max_results_off=1, max_results_usda=1),
            ]
        )
    )
    csv_text = result.render_csv()
    assert csv_text.index("# query: milk") < csv_text.index("# query: cheese")
    assert csv_text.count("# query: ") == 2


def test_batch_json_renders_one_query_section_per_input() -> None:
    off = _FakeDb("off", [_off_item("1")])
    usda = _FakeDb("usda", [_usda_item("usda:10")])
    result = asyncio.run(
        _facade(off, usda).lookup(
            [
                LookupQuery(query="milk", max_results_off=1, max_results_usda=1),
                LookupQuery(query="cheese", max_results_off=1, max_results_usda=1),
            ]
        )
    )
    json_text = result.render_json()
    assert json_text.index('"query":"milk"') < json_text.index('"query":"cheese"')
    assert json_text.count('"query":') == 2


def test_per_source_limit_of_zero_skips_that_source() -> None:
    off = _FakeDb("off", [_off_item(str(i)) for i in range(5)])
    usda = _FakeDb("usda", [_usda_item(f"usda:{i}") for i in range(5)])

    staple_query = LookupQuery(query="carrot", max_results_off=0, max_results_usda=3)
    staple = asyncio.run(_facade(off, usda).lookup([staple_query]))
    assert staple.queries[0].results.open_food_facts == []
    assert len(_usda(staple)) == 3

    branded_query = LookupQuery(query="lays classic", max_results_off=5, max_results_usda=0)
    branded = asyncio.run(_facade(off, usda).lookup([branded_query]))
    assert len(_off(branded)) == 5
    assert branded.queries[0].results.usda == []


# --- open() honours usage settings -----------------------------------------


def test_open_skips_sources_disabled_by_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DA_OFF_USAGE", "disabled")
    monkeypatch.setenv("DA_USDA_USAGE", "disabled")
    get_settings.cache_clear()
    with FoodDb.open() as db:
        result = asyncio.run(db.lookup([LookupQuery(query="milk", max_results_off=3, max_results_usda=3)]))
        assert result == BatchLookupResult(queries=[QueryLookupResult(query="milk")])
        with pytest.raises(OFFUnknownFoodCodeError):
            db.get_food("off:5901234")


def test_open_raises_loudly_when_enabled_db_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DA_OFF_USAGE", "full")
    monkeypatch.setenv("DA_USDA_USAGE", "disabled")
    monkeypatch.setenv("DA_OFF_DB", str(tmp_path / "missing.duckdb"))
    get_settings.cache_clear()
    with pytest.raises(FileNotFoundError):
        FoodDb.open()


# --- nutrient display rounding ----------------------------------------------


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("energy_kcal", 8.300000190734863, Decimal("8.3")),
        ("energy_kcal", 1399.999976158142, Decimal("1400.0")),
        ("energy_kcal", 1017.4000244140625, Decimal("1017.4")),
        ("sodium_mg", 560.0000023841858, Decimal("560")),
        ("iron_mg", 2.7100000381469727, Decimal("2.710")),
        ("vitamin_d_ug", 0.015, Decimal("0.015")),
        ("protein_g", None, None),
    ],
)
def test_hit_nutrients_round_per_field_precision(field: str, raw: float | None, expected: Decimal | None) -> None:
    hit = OFFHit.model_validate({"code": "off:1", "name": "Bread", field: raw})
    assert getattr(hit, field) == expected


def test_float32_artifact_renders_clean_in_csv_and_json() -> None:
    item = OFFItem(code="1", product_name="Bread", energy_kcal_in_100g=8.300000190734863)
    off = _FakeDb("off", [item])
    result = asyncio.run(_facade(off, None).lookup([LookupQuery(query="bread", max_results_off=1, max_results_usda=0)]))

    hit = result.queries[0].results.open_food_facts[0]
    assert hit.energy_kcal == Decimal("8.3")
    assert hit.code == "off:1"
    assert "8.3" in hit.csv_row()
    assert "8.3000001" not in result.render_csv()

    json_text = result.render_json()
    assert '"energy_kcal":8.3' in json_text
    assert "8.300000190734863" not in json_text


def test_empty_hit_drops_none_nutrients_from_json() -> None:
    hit = OFFHit(code="off:1", name="Bread")
    result = BatchLookupResult(queries=[QueryLookupResult(query="bread", results=LookupResult(open_food_facts=[hit]))])
    assert '"energy_kcal"' not in result.render_json()
