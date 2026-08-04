"""Local USDA FoodData Central database (DuckDB), the generic-food source.

The whole-food complement to `OffFoodDb`: Foundation Foods and SR Legacy, the
generic/reference foods OFF's branded Polish catalogue largely lacks. Same
hybrid retrieval (BM25 + embedding, fused with RRF) so the two databases can be
searched identically and their results blended by the agent tools.

Codes are the FDC id prefixed with `usda:` so they never collide with OFF
barcodes and eval hydration can route each code back to the DB that owns it.
Rows are returned as 1:1 `USDAItem`s; nutrient units are already canonicalized
at build time.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb

from dietary_advisor.config import FoodDbUsage, get_settings
from dietary_advisor.food_db.embeddings import embed_query
from dietary_advisor.food_db.errors import USDAUnknownFoodCodeError
from dietary_advisor.food_db.fusion import reciprocal_rank_fusion
from dietary_advisor.food_db.models import Row, USDAItem

log = logging.getLogger(__name__)

_CODE_PREFIX = "usda:"

_SELECT_COLUMNS = (
    "fdc_id, description, category, "
    "energy_kcal_in_100g, proteins_g_in_100g, carbohydrates_g_in_100g, fat_g_in_100g, "
    "saturated_fat_g_in_100g, fiber_g_in_100g, sugars_g_in_100g, "
    "sodium_mg_in_100g, potassium_mg_in_100g, calcium_mg_in_100g, iron_mg_in_100g, "
    "vitamin_c_mg_in_100g, vitamin_d_ug_in_100g, cholesterol_mg_in_100g"
)


def to_code(fdc_id: int | str) -> str:
    """Render an FDC id as a runtime `usda:<fdc_id>` code."""
    return f"{_CODE_PREFIX}{fdc_id}"


def is_usda_code(code: str) -> bool:
    return code.startswith(_CODE_PREFIX)


def _fdc_id(code: str) -> int:
    raw = code[len(_CODE_PREFIX) :] if is_usda_code(code) else code
    try:
        return int(raw)
    except ValueError as exc:
        raise USDAUnknownFoodCodeError(code, f"not a USDA code: {code!r}") from exc


def get_usda_item_name(item: USDAItem | Mapping[str, Any]) -> str:
    get = item.get if isinstance(item, Mapping) else lambda k: getattr(item, k, None)
    description = get("description")
    if description:
        return str(description)
    fdc_id = get("fdc_id")
    return to_code(fdc_id) if fdc_id is not None else "unknown"


def _row_to_usda_item(row: Mapping[str, Any]) -> USDAItem:
    return USDAItem.model_validate(dict(row))


class UsdaFoodDb:
    """Read-only accessor over the local USDA FoodData Central DuckDB.

    Opened read-only so it can never mutate the artifact built by `setup` and so
    several instances can share the file across the pipeline and eval harness.
    """

    def __init__(self, db_path: Path | None = None, usage: FoodDbUsage | None = None) -> None:
        settings = get_settings()
        path = db_path or settings.usda_db
        if not Path(path).exists():
            raise FileNotFoundError(
                f"USDA food DB not found at {path}. Build it first with `just setup` (or `python -m setup`).",
            )
        self._con = duckdb.connect(str(path), read_only=True)
        self._embedding_dim = settings.off_embedding_dim
        self._usage = usage or settings.usda_usage
        try:
            self._con.execute("INSTALL vss")
            self._con.execute("LOAD vss")
        except duckdb.Error as exc:
            log.warning("Could not load DuckDB VSS extension; semantic search will brute-force scan: %s", exc)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> UsdaFoodDb:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _rows(self, sql: str, params: list[Any]) -> list[Row]:
        cur = self._con.execute(sql, params)
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def _bm25_rows(self, query: str, limit: int) -> list[Row]:
        """Lexical (BM25) candidates from the full-text index built by `setup`."""
        return self._rows(
            f"SELECT * FROM ("  # noqa: S608 (static column list; params are bound)
            f"  SELECT {_SELECT_COLUMNS}, fts_main_foods.match_bm25(fdc_id, ?) AS score FROM foods"
            f") WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
            [query, limit],
        )

    def _semantic_rows(self, query: str, limit: int) -> list[Row]:
        """Nearest foods to `query` by embedding cosine distance.

        Degrades to an empty list (rather than raising) when the DB predates the
        `embedding` column or the embedder is unavailable, so lexical search
        alone still answers the query.
        """
        try:
            vector = embed_query(query)
            return self._rows(
                f"SELECT {_SELECT_COLUMNS} FROM foods "  # noqa: S608 (static column list; params are bound)
                f"WHERE embedding IS NOT NULL "
                f"ORDER BY array_cosine_distance(embedding, ?::FLOAT[{self._embedding_dim}]) LIMIT ?",
                [vector, limit],
            )
        except Exception as exc:  # noqa: BLE001 (semantic channel is best-effort)
            log.warning("Semantic search unavailable, using lexical results only: %s", exc)
            return []

    def search(self, query: str, limit: int = 5) -> list[USDAItem]:
        """Return up to `limit` foods best matching `query`, ranked by relevance.

        Channel selection follows `self._usage`, identical to the OFF reader:
        `only_bm25` / `only_semantic` run a single channel, `full` merges the
        DuckDB BM25 index and embedding cosine similarity via reciprocal rank
        fusion.
        """
        query = query.strip()
        if not query or self._usage is FoodDbUsage.DISABLED:
            return []
        if self._usage is FoodDbUsage.ONLY_BM25:
            rows = self._bm25_rows(query, limit)
        elif self._usage is FoodDbUsage.ONLY_SEMANTIC:
            rows = self._semantic_rows(query, limit)
        else:
            candidates = max(limit * 4, 20)
            bm25 = self._bm25_rows(query, candidates)
            semantic = self._semantic_rows(query, candidates)
            rows = reciprocal_rank_fusion([bm25, semantic], key=lambda r: r["fdc_id"])[:limit]
        return [_row_to_usda_item(r) for r in rows]

    def get_food(self, code: str) -> USDAItem:
        """Return the food with `usda:<fdc_id>` `code`, or raise `USDAUnknownFoodCodeError` if absent."""
        rows = self._rows(
            f"SELECT {_SELECT_COLUMNS} FROM foods WHERE fdc_id = ? LIMIT 1",  # noqa: S608 (static columns)
            [_fdc_id(code)],
        )
        if not rows:
            raise USDAUnknownFoodCodeError(code, f"unknown USDA food code: {code!r}")
        return _row_to_usda_item(rows[0])
