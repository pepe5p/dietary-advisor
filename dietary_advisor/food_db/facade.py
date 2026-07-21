"""`FoodDb`: the single food-database entry point for the rest of the system.

Callers (the nutrition agent, the evaluation harness) work with foods, not
with the fact that there are two underlying DuckDB catalogues. This facade
owns opening, searching, code resolution, and closing of both the branded
Open Food Facts reader and the generic USDA reader, and hides the routing.

A source participates only when its `usage` is not `disabled`; both artifacts
are otherwise assumed to exist, so an enabled-but-missing DB raises loudly at
`open` rather than degrading silently.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from collections.abc import Sequence
from typing import ClassVar, Protocol, Self

from pydantic import BaseModel, Field

from dietary_advisor.config import FoodDbUsage, get_settings, Settings
from dietary_advisor.food_db.errors import OFFUnknownFoodCodeError, USDAUnknownFoodCodeError
from dietary_advisor.food_db.models import OFFItem, USDAItem
from dietary_advisor.food_db.off_food_db import get_off_item_name, OffFoodDb
from dietary_advisor.food_db.usda_food_db import get_usda_item_name, is_usda_code, to_code, UsdaFoodDb

log = logging.getLogger(__name__)

# Cap the number of distinct queries a single batch lookup will run, so one
# tool call can't fan out into an unbounded number of DB searches.
_MAX_BATCH_QUERIES = 20

# Ingredient lists can run to hundreds of characters; cap the search-result
# preview so a batch of hits doesn't crowd out the prompt (full text is still
# available at hydration, straight from the DB row).
_INGREDIENTS_SUMMARY_MAX_CHARS = 200

# Totaller nutrients shown on every hit, named with their canonical unit suffix.
_SHARED_NUTRIENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("energy_kcal", "energy_kcal_in_100g"),
    ("protein_g", "proteins_g_in_100g"),
    ("carbs_g", "carbohydrates_g_in_100g"),
    ("fat_g", "fat_g_in_100g"),
    ("saturated_fat_g", "saturated_fat_g_in_100g"),
    ("fiber_g", "fiber_g_in_100g"),
    ("sugar_g", "sugars_g_in_100g"),
    ("sodium_mg", "sodium_mg_in_100g"),
    ("potassium_mg", "potassium_mg_in_100g"),
    ("calcium_mg", "calcium_mg_in_100g"),
    ("iron_mg", "iron_mg_in_100g"),
    ("vitamin_c_mg", "vitamin_c_mg_in_100g"),
    ("vitamin_d_ug", "vitamin_d_ug_in_100g"),
    ("cholesterol_mg", "cholesterol_mg_in_100g"),
)

_OFF_EXTRA_NUTRIENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("energy_kj", "energy_kj_in_100g"),
    ("salt_g", "salt_g_in_100g"),
)


class LookupQuery(BaseModel):
    """One ingredient search with its own per-source result cap."""

    query: str
    max_results_off: int = Field(default=2, ge=0, le=15)
    max_results_usda: int = Field(default=2, ge=0, le=10)


class _NutrientHit(BaseModel):
    """Per-100g nutrients every hit carries (canonical units, already in the DB)."""

    energy_kcal: float | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    saturated_fat_g: float | None = None
    fiber_g: float | None = None
    sugar_g: float | None = None
    sodium_mg: float | None = None
    potassium_mg: float | None = None
    calcium_mg: float | None = None
    iron_mg: float | None = None
    vitamin_c_mg: float | None = None
    vitamin_d_ug: float | None = None
    cholesterol_mg: float | None = None


def _nutrient_kwargs(item: OFFItem | USDAItem, fields: tuple[tuple[str, str], ...]) -> dict[str, float | None]:
    return {hit_key: getattr(item, col) for hit_key, col in fields}


def _truncate(text: str | None) -> str | None:
    if text is None or len(text) <= _INGREDIENTS_SUMMARY_MAX_CHARS:
        return text
    return text[:_INGREDIENTS_SUMMARY_MAX_CHARS].rstrip() + "..."


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:g}"


class OFFHit(_NutrientHit):
    """A single Open Food Facts search hit as surfaced to the agent."""

    _CSV_CONTEXT: ClassVar[tuple[str, ...]] = ("code", "name", "brands", "categories", "ingredients_text")
    _CSV_NUTRIENTS: ClassVar[tuple[str, ...]] = tuple(
        k for k, _ in _SHARED_NUTRIENT_FIELDS + _OFF_EXTRA_NUTRIENT_FIELDS
    )

    code: str
    name: str
    brands: str | None = None
    categories: str | None = None
    ingredients_text: str | None = None
    energy_kj: float | None = None
    salt_g: float | None = None

    @classmethod
    def create_from_off_item(cls, item: OFFItem) -> Self:
        return cls(
            code=item.code,
            name=get_off_item_name(item),
            brands=item.brands,
            categories=item.categories,
            ingredients_text=_truncate(item.ingredients_text),
            **_nutrient_kwargs(item, _SHARED_NUTRIENT_FIELDS + _OFF_EXTRA_NUTRIENT_FIELDS),
        )

    def csv_row(self) -> list[str]:
        context = [self.code, self.name, self.brands or "", self.categories or "", self.ingredients_text or ""]
        nutrients = [_fmt(getattr(self, key)) for key in self._CSV_NUTRIENTS]
        return context + nutrients


class USDAHit(_NutrientHit):
    """A single USDA search hit as surfaced to the agent."""

    _CSV_CONTEXT: ClassVar[tuple[str, ...]] = ("code", "name", "category")
    _CSV_NUTRIENTS: ClassVar[tuple[str, ...]] = tuple(k for k, _ in _SHARED_NUTRIENT_FIELDS)

    code: str
    name: str
    category: str | None = None

    @classmethod
    def create_from_usda_item(cls, item: USDAItem) -> Self:
        return cls(
            code=to_code(item.fdc_id),
            name=get_usda_item_name(item),
            category=item.category,
            **_nutrient_kwargs(item, _SHARED_NUTRIENT_FIELDS),
        )

    def csv_row(self) -> list[str]:
        context = [self.code, self.name, self.category or ""]
        nutrients = [_fmt(getattr(self, key)) for key in self._CSV_NUTRIENTS]
        return context + nutrients


def _csv_block(title: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return f"# {title}\n(no hits)"
    buf = io.StringIO()
    buf.write(f"# {title}\n")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().rstrip()


class LookupResult(BaseModel):
    """Search hits grouped by source; render to CSV for the LLM tool result."""

    open_food_facts: list[OFFHit] = Field(default_factory=list)
    usda: list[USDAHit] = Field(default_factory=list)

    def render_csv(self) -> str:
        """Compact CSV (one block per source) for the nutrition-agent tool result."""
        off_header = list(OFFHit._CSV_CONTEXT) + list(OFFHit._CSV_NUTRIENTS)
        usda_header = list(USDAHit._CSV_CONTEXT) + list(USDAHit._CSV_NUTRIENTS)
        off_block = _csv_block(
            "open_food_facts",
            off_header,
            [h.csv_row() for h in self.open_food_facts],
        )
        usda_block = _csv_block(
            "usda",
            usda_header,
            [h.csv_row() for h in self.usda],
        )
        return f"{off_block}\n\n{usda_block}"


class _Searchable[ItemT](Protocol):
    def search(self, query: str, limit: int = ...) -> list[ItemT]: ...


def _off_hits_summary(items: list[OFFItem]) -> list[OFFHit]:
    return [OFFHit.create_from_off_item(it) for it in items]


def _usda_hits_summary(items: list[USDAItem]) -> list[USDAHit]:
    return [USDAHit.create_from_usda_item(it) for it in items]


def _safe_search[ItemT](db: _Searchable[ItemT], query: str, limit: int) -> list[ItemT]:
    try:
        return list(db.search(query, limit=limit))
    except Exception as exc:  # noqa: BLE001 (each source is best-effort)
        log.warning("Food DB search failed for %r: %s", query, exc)
        return []


async def _search_source[ItemT](
    db: _Searchable[ItemT] | None,
    queries: Sequence[tuple[str, int]],
) -> list[list[ItemT]]:
    """Run each query against one DB off the event loop, one hit list per query.

    Each query carries its own limit; a limit of 0 (or a disabled source)
    skips the DB call and yields an empty list for that query. Results stay
    grouped per query (index-aligned with `queries`) so the caller can report
    per-query hit counts.
    """
    hits_per_query: list[list[ItemT]] = []
    for query, limit in queries:
        if db is None or limit <= 0:
            hits_per_query.append([])
        else:
            hits_per_query.append(await asyncio.to_thread(_safe_search, db, query, limit))
    return hits_per_query


class FoodDb:
    """Facade over the OFF and USDA readers behind one food-lookup surface.

    Either source may be `None` (its `usage` is `disabled`); in that case it
    contributes no search hits and lookups of its codes raise `UnknownFoodCodeError`.
    """

    def __init__(self, off_db: OffFoodDb | None = None, usda_db: UsdaFoodDb | None = None) -> None:
        self._off = off_db
        self._usda = usda_db

    @classmethod
    def open(cls, settings: Settings | None = None) -> FoodDb:
        """Open every enabled food DB, failing loudly if its artifact is missing.

        A source is opened iff its usage is not `disabled`; opening a missing DB
        raises from the reader's `__init__` instead of being swallowed, so a
        forgotten `just setup` surfaces immediately rather than as silently
        absent lookups.
        """
        settings = settings or get_settings()
        off = OffFoodDb(settings.off_db) if settings.off_usage is not FoodDbUsage.DISABLED else None
        usda = UsdaFoodDb(settings.usda_db) if settings.usda_usage is not FoodDbUsage.DISABLED else None
        return cls(off, usda)

    async def lookup(self, queries: Sequence[LookupQuery]) -> LookupResult:
        """Search both sources concurrently, returning hits grouped by source.

        Each `LookupQuery` caps hits per source independently, so the caller
        (the LLM) can weight a query towards OFF, towards USDA, or skip a
        source entirely (limit 0) rather than searching both identically.

        The two sources run in parallel (each is a synchronous DuckDB reader
        driven off the event loop) and their results stay separated under
        distinct keys so the caller can tell a branded Polish product from a
        generic USDA food. A disabled source contributes an empty list.
        """
        queries = list(queries)[:_MAX_BATCH_QUERIES]
        off_per_query, usda_per_query = await asyncio.gather(
            _search_source(self._off, [(q.query, q.max_results_off) for q in queries]),
            _search_source(self._usda, [(q.query, q.max_results_usda) for q in queries]),
        )
        for q, off_hits, usda_hits in zip(queries, off_per_query, usda_per_query, strict=True):
            log.debug(
                "lookup query %r (off_limit=%d, usda_limit=%d): %d combined hit(s), %d from USDA, %d from OFF",
                q.query,
                q.max_results_off,
                q.max_results_usda,
                len(off_hits) + len(usda_hits),
                len(usda_hits),
                len(off_hits),
            )
        off_flat = [hit for query_hits in off_per_query for hit in query_hits]
        usda_flat = [hit for query_hits in usda_per_query for hit in query_hits]
        return LookupResult(
            open_food_facts=_off_hits_summary(off_flat),
            usda=_usda_hits_summary(usda_flat),
        )

    def get_food(self, code: str) -> OFFItem | USDAItem:
        """Resolve `code` to its read model, routing by prefix to the owning source."""
        if is_usda_code(code):
            if self._usda is None:
                raise USDAUnknownFoodCodeError(f"USDA food DB unavailable for code: {code!r}")
            return self._usda.get_food(code)
        if self._off is None:
            raise OFFUnknownFoodCodeError(f"OFF food DB unavailable for code: {code!r}")
        return self._off.get_food(code)

    def close(self) -> None:
        for db in (self._off, self._usda):
            if db is not None:
                db.close()

    def __enter__(self) -> FoodDb:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
