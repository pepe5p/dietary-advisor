"""REPL helpers for exploring the OFF and USDA food DBs.

Includes per-channel (`_bm25`/`_semantic`) search variants that reach into
each reader's private methods to bypass the hybrid RRF fusion - useful for
debugging a query where fusion buries one channel's best hit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import duckdb
from rich.pretty import pprint
from rich.table import Table

from dietary_advisor.cli.rendering import format_extra_nutrients
from dietary_advisor.config import get_settings
from dietary_advisor.food_db import FoodDb, LookupQuery, LookupResult, OffFoodDb, OFFItem, UsdaFoodDb, USDAItem
from dietary_advisor.food_db.models import Row
from dietary_advisor.food_db.nutrients import nutrients_from_row
from dietary_advisor.food_db.off_food_db import _row_to_off_item, get_off_item_name
from dietary_advisor.food_db.usda_food_db import _fdc_id, _row_to_usda_item, get_usda_item_name, to_code
from dietary_advisor.totaller.nutrition import canonical_unit, NutrientName
from repl.manual import console, print_manual
from setup.settings import get_setup_settings

__all__ = [
    "get_off_item",
    "get_off_record",
    "get_parquet_record",
    "get_usda_item",
    "get_usda_record",
    "lookup",
    "lookup_csv",
    "pdict",
    "pfi",
    "search_off",
    "search_off_bm25",
    "search_off_semantic",
    "search_usda",
    "search_usda_bm25",
    "search_usda_semantic",
    "sql_off",
    "sql_usda",
]

# The two food_db read models don't share a base class (see food_db/models.py).
type ReadModel = OFFItem | USDAItem

_MACRO_COLUMNS: tuple[NutrientName, ...] = (
    NutrientName.ENERGY_KCAL,
    NutrientName.PROTEIN_G,
    NutrientName.CARBS_G,
    NutrientName.FAT_G,
)


def _execute_sql(con: duckdb.DuckDBPyConnection, sql: str) -> list[Row]:
    cur = con.execute(sql)
    if cur.description is None:
        return []
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def _fetch_record(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any], code: str) -> Row:
    cur = con.execute(sql, params)
    columns = [d[0] for d in cur.description]
    row = cur.fetchone()
    if row is None:
        raise KeyError(f"unknown product code: {code!r}")
    return dict(zip(columns, row, strict=True))


def _channel[ReadModelT: ReadModel](
    db: OffFoodDb | UsdaFoodDb,
    method: str,
    query: str,
    limit: int,
    mapper: Callable[[Row], ReadModelT],
) -> list[ReadModelT]:
    """Run one retrieval channel (`_bm25_rows` / `_semantic_rows`) in isolation.

    Bypasses the reader's hybrid `search()` (which fuses both channels via
    RRF) so each channel's raw ranking can be inspected on its own - e.g. to
    debug a query where fusion buries a channel's best hit.
    """
    query = query.strip()
    if not query:
        return []
    with db as conn:
        rows = getattr(conn, method)(query, limit)
        return [mapper(r) for r in rows]


def search_off(query: str, limit: int = 5) -> list[OFFItem]:
    with OffFoodDb() as off_food_db:
        return off_food_db.search(query=query, limit=limit)


def search_off_bm25(query: str, limit: int = 5) -> list[OFFItem]:
    return _channel(OffFoodDb(), "_bm25_rows", query, limit, _row_to_off_item)


def search_off_semantic(query: str, limit: int = 5) -> list[OFFItem]:
    return _channel(OffFoodDb(), "_semantic_rows", query, limit, _row_to_off_item)


def get_off_item(code: str) -> OFFItem:
    with OffFoodDb() as off_food_db:
        return off_food_db.get_food(code=code)


def get_off_record(code: str) -> Row:
    """Return the full, unprocessed OFF record for `code` from the slimmed `.duckdb`."""
    con = duckdb.connect(str(get_settings().off_db), read_only=True)
    try:
        return _fetch_record(con, "SELECT * FROM products WHERE code = ? LIMIT 1", [code], code)
    finally:
        con.close()


def sql_off(sql: str) -> list[Row]:
    con = duckdb.connect(str(get_settings().off_db), read_only=True)
    try:
        return _execute_sql(con, sql)
    finally:
        con.close()


def search_usda(query: str, limit: int = 5) -> list[USDAItem]:
    with UsdaFoodDb() as usda_food_db:
        return usda_food_db.search(query=query, limit=limit)


def search_usda_bm25(query: str, limit: int = 5) -> list[USDAItem]:
    return _channel(UsdaFoodDb(), "_bm25_rows", query, limit, _row_to_usda_item)


def search_usda_semantic(query: str, limit: int = 5) -> list[USDAItem]:
    return _channel(UsdaFoodDb(), "_semantic_rows", query, limit, _row_to_usda_item)


def get_usda_item(code: str) -> USDAItem:
    with UsdaFoodDb() as usda_food_db:
        return usda_food_db.get_food(code=code)


def get_usda_record(code: str) -> Row:
    """Return the full, unprocessed USDA record for `code` (bare fdc_id or `usda:<fdc_id>`)."""
    con = duckdb.connect(str(get_settings().usda_db), read_only=True)
    try:
        return _fetch_record(con, "SELECT * FROM foods WHERE fdc_id = ? LIMIT 1", [_fdc_id(code)], code)
    finally:
        con.close()


def sql_usda(sql: str) -> list[Row]:
    con = duckdb.connect(str(get_settings().usda_db), read_only=True)
    try:
        return _execute_sql(con, sql)
    finally:
        con.close()


def lookup(query: str, off_limit: int = 5, usda_limit: int = 5) -> LookupResult:
    """Run the whole `FoodDb.lookup` facade path for one query, grouped by source."""
    with FoodDb.open() as food_db:
        query_obj = LookupQuery(query=query, max_results_off=off_limit, max_results_usda=usda_limit)
        return asyncio.run(food_db.lookup([query_obj]))


def lookup_csv(query: str, off_limit: int = 5, usda_limit: int = 5) -> str:
    """Same as `lookup`, but return the CSV text the nutrition agent tools see."""
    return lookup(query, off_limit=off_limit, usda_limit=usda_limit).render_csv()


def get_parquet_record(code: str) -> Row:
    """Return the full, unprocessed OFF record for `code` from the raw Parquet export."""
    parquet = get_setup_settings().off_raw_parquet
    con = duckdb.connect()
    try:
        return _fetch_record(
            con,
            f"SELECT * FROM read_parquet('{parquet}') WHERE code = ? LIMIT 1",  # noqa: S608
            [code],
            code,
        )
    finally:
        con.close()


_MAX_INLINE_SEQ = 16


def _shorten_long_sequences(value: Any) -> Any:
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, dict):
        return {k: _shorten_long_sequences(v) for k, v in value.items()}
    if hasattr(value, "__len__") and len(value) > _MAX_INLINE_SEQ:
        return f"[... {len(value)} values ...]"
    if isinstance(value, list | tuple):
        return type(value)(_shorten_long_sequences(v) for v in value)
    return value


def pdict(record: Row) -> None:
    pprint(_shorten_long_sequences(record))


def _item_name(item: ReadModel) -> str:
    return get_off_item_name(item) if isinstance(item, OFFItem) else get_usda_item_name(item)


def _item_code(item: ReadModel) -> str:
    return item.code if isinstance(item, OFFItem) else to_code(item.fdc_id)


def _food_items_table(food_items: list[ReadModel]) -> Table:
    table = Table(title="Food items", show_lines=True, header_style="bold cyan")
    table.add_column("Name", style="bold", overflow="fold")
    table.add_column("Code", style="dim")
    for nutrient in _MACRO_COLUMNS:
        table.add_column(f"{nutrient.value} ({canonical_unit(nutrient)})", justify="right")
    table.add_column("Other nutrients", overflow="fold")

    for item in food_items:
        nutrients = nutrients_from_row(item)
        macros = [f"{nutrients[n]:.1f}" if n in nutrients else "-" for n in _MACRO_COLUMNS]
        extras = format_extra_nutrients(nutrients) or "-"
        table.add_row(_item_name(item), _item_code(item), *macros, extras)
    return table


def _food_item_detail_table(food_item: ReadModel) -> Table:
    name = _item_name(food_item)
    table = Table(title=f"Food item: {name}", show_lines=True, header_style="bold cyan")
    table.add_column("Field", style="bold", overflow="fold")
    table.add_column("Value", overflow="fold")

    table.add_row("name", name)
    table.add_row("code", _item_code(food_item))

    if isinstance(food_item, OFFItem):
        table.add_row("brands", food_item.brands or "-")
        table.add_row("categories", food_item.categories or "-")
        table.add_row("compared_to_category", food_item.compared_to_category or "-")
        table.add_row("ingredients_text", food_item.ingredients_text or "-")
        table.add_row("allergens_tags", ", ".join(food_item.allergens_tags) or "-")
        table.add_row("labels_tags", ", ".join(food_item.labels_tags) or "-")
    else:
        table.add_row("category", food_item.category or "-")

    nutrients = nutrients_from_row(food_item)
    for nutrient in NutrientName:
        amount = nutrients.get(nutrient)
        value = f"{amount:.3g} {canonical_unit(nutrient)}" if amount is not None else "[dim]-[/dim]"
        table.add_row(f"{nutrient.value} / 100g", value)
    return table


def pfi(items: ReadModel | list[ReadModel]) -> None:
    if isinstance(items, OFFItem | USDAItem):
        console.print(_food_item_detail_table(items))
        return
    if not items:
        console.print("[yellow]No food items.[/yellow]")
        return
    console.print(_food_items_table(items))


_MANUAL: tuple[tuple[str, str], ...] = (
    ("search_off(query, limit=5)", "Search OFF (hybrid BM25 + semantic) -> list[OFFItem]."),
    ("search_off_bm25(query, limit=5)", "Search OFF, lexical (BM25) channel only."),
    ("search_off_semantic(query, limit=5)", "Search OFF, embedding (semantic) channel only."),
    ("get_off_item(code)", "Get one product as an OFFItem."),
    ("get_off_record(code)", "Get the full raw row (dict) from the OFF duckdb."),
    ("sql_off(sql)", "Run raw SQL against the OFF duckdb -> list[Row]."),
    ("get_parquet_record(code)", "Get the full raw row (dict) from the OFF parquet export."),
    ("search_usda(query, limit=5)", "Search USDA (hybrid BM25 + semantic) -> list[USDAItem]."),
    ("search_usda_bm25(query, limit=5)", "Search USDA, lexical (BM25) channel only."),
    ("search_usda_semantic(query, limit=5)", "Search USDA, embedding (semantic) channel only."),
    ("get_usda_item(code)", "Get one food as a USDAItem."),
    ("get_usda_record(code)", "Get the full raw row (dict) from the USDA duckdb."),
    ("sql_usda(sql)", "Run raw SQL against the USDA duckdb -> list[Row]."),
    ("lookup(query, off_limit=5, usda_limit=5)", "Run the full dual-source FoodDb.lookup -> LookupResult."),
    ("lookup_csv(query, off_limit=5, usda_limit=5)", "Same as lookup, but as the CSV the agent tools return."),
    ("pfi(items)", "Rich-print an OFFItem/USDAItem or a list of them."),
    ("pdict(record)", "Rich pretty-print a raw record dict."),
)

print_manual("food DB helpers", _MANUAL)
