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
import logging
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from dietary_advisor.config import FoodDbUsage, get_settings, Settings
from dietary_advisor.food_db.errors import OFFUnknownFoodCodeError, USDAUnknownFoodCodeError
from dietary_advisor.food_db.models import OFFItem, USDAItem
from dietary_advisor.food_db.off_food_db import OffFoodDb
from dietary_advisor.food_db.usda_food_db import is_usda_code, UsdaFoodDb
from dietary_advisor.totaller.nutrition import NutrientName

log = logging.getLogger(__name__)

# Cap the number of distinct queries a single batch lookup will run, so one
# tool call can't fan out into an unbounded number of DB searches.
_MAX_BATCH_QUERIES = 20

# The LLM only needs enough per-100g context to allocate grams sensibly (the
# Totaller re-verifies everything deterministically after), so the four
# macros are enough - no point spending tokens on every micronutrient.
_MACRO_KEYS: tuple[tuple[str, NutrientName], ...] = (
    ("kcal_per_100g", NutrientName.ENERGY_KCAL),
    ("protein_g_per_100g", NutrientName.PROTEIN_G),
    ("carbs_g_per_100g", NutrientName.CARBS_G),
    ("fat_g_per_100g", NutrientName.FAT_G),
)

# Ingredient lists can run to hundreds of characters; cap the search-result
# preview so a batch of hits doesn't crowd out the prompt (full text is still
# available at hydration, straight from the DB row).
_INGREDIENTS_SUMMARY_MAX_CHARS = 200


class LookupQuery(BaseModel):
    """One ingredient search with its own per-source result cap."""

    query: str
    max_results_off: int = Field(default=2, ge=0, le=15)
    max_results_usda: int = Field(default=2, ge=0, le=10)


class _MacroHit(BaseModel):
    """The four per-100g macros every hit carries, shared by both sources."""

    kcal_per_100g: float = 0.0
    protein_g_per_100g: float = 0.0
    carbs_g_per_100g: float = 0.0
    fat_g_per_100g: float = 0.0


class OFFHit(_MacroHit):
    """A single Open Food Facts search hit as surfaced to the agent."""

    code: str
    name: str
    brands: str | None = None
    categories: str | None = None
    ingredients_text: str | None = None
    tags: list[str] = Field(default_factory=list)


class USDAHit(_MacroHit):
    """A single USDA search hit as surfaced to the agent."""

    code: str
    name: str
    category: str | None = None


class LookupResult(BaseModel):
    """Search hits grouped by source, the return shape of the agent lookup tools."""

    open_food_facts: list[OFFHit] = Field(default_factory=list)
    usda: list[USDAHit] = Field(default_factory=list)


class _Searchable[ItemT](Protocol):
    def search(self, query: str, limit: int = ...) -> list[ItemT]: ...


def _macro_fields(item: OFFItem | USDAItem) -> dict[str, float]:
    return {key: item.nutrients_per_100g.get(nutrient, 0.0) for key, nutrient in _MACRO_KEYS}


def _truncate(text: str | None) -> str | None:
    if text is None or len(text) <= _INGREDIENTS_SUMMARY_MAX_CHARS:
        return text
    return text[:_INGREDIENTS_SUMMARY_MAX_CHARS].rstrip() + "..."


def _off_hits_summary(items: list[OFFItem]) -> list[OFFHit]:
    return [
        OFFHit(
            code=it.code,
            name=it.name,
            brands=it.brands,
            categories=it.categories,
            ingredients_text=_truncate(it.ingredients_text),
            tags=it.tags,
            **_macro_fields(it),
        )
        for it in items
    ]


def _usda_hits_summary(items: list[USDAItem]) -> list[USDAHit]:
    return [
        USDAHit(
            code=it.code,
            name=it.name,
            category=it.category,
            **_macro_fields(it),
        )
        for it in items
    ]


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
