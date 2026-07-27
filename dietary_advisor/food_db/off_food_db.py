"""Local Open Food Facts product database (DuckDB), the branded-food source.

The branded complement to `UsdaFoodDb`: Polish packaged products. Same hybrid
retrieval (BM25 + embedding, fused with RRF) so the two databases can be
searched identically and their results blended by the agent tools.

Codes are the barcode prefixed with `off:` so they never collide with USDA
FDC ids and eval hydration can route each code back to the DB that owns it.
Rows are returned as 1:1 `OFFItem`s; nutrient units are already canonicalized
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
from dietary_advisor.food_db.errors import OFFUnknownFoodCodeError
from dietary_advisor.food_db.fusion import reciprocal_rank_fusion as _reciprocal_rank_fusion
from dietary_advisor.food_db.models import OFFItem, Row

log = logging.getLogger(__name__)

_CODE_PREFIX = "off:"

# Every stored column except the embedding vector (never needed at read time).
_SELECT_COLUMNS = (
    "code, product_name, product_name_pl, ingredients_text, "
    "brands, brands_tags, quantity, serving_size, serving_quantity, "
    "product_quantity, product_quantity_unit, nutrition_data_per, "
    "categories, categories_tags, compared_to_category, "
    "labels_tags, allergens_tags, traces_tags, additives_tags, "
    "nova_group, nutriscore_grade, nutriscore_score, "
    "energy_kcal_in_100g, energy_kj_in_100g, proteins_g_in_100g, carbohydrates_g_in_100g, "
    "sugars_g_in_100g, fat_g_in_100g, saturated_fat_g_in_100g, fiber_g_in_100g, salt_g_in_100g, "
    "sodium_mg_in_100g, potassium_mg_in_100g, calcium_mg_in_100g, iron_mg_in_100g, "
    "vitamin_c_mg_in_100g, vitamin_d_ug_in_100g, cholesterol_mg_in_100g"
)


def to_code(barcode: str) -> str:
    """Render a barcode as a runtime `off:<barcode>` code."""
    return f"{_CODE_PREFIX}{barcode}"


def is_off_code(code: str) -> bool:
    return code.startswith(_CODE_PREFIX)


def _barcode(code: str) -> str:
    return code[len(_CODE_PREFIX) :] if is_off_code(code) else code


def get_off_item_name(item: OFFItem | Mapping[str, Any]) -> str:
    """Display name fallback used by hits and hydration."""
    get = item.get if isinstance(item, Mapping) else lambda k: getattr(item, k, None)
    return get("product_name") or get("product_name_pl") or get("brands") or "unknown"


def _row_to_off_item(row: Mapping[str, Any]) -> OFFItem:
    return OFFItem.model_validate(dict(row))


class OffFoodDb:
    """Read-only accessor over the local Open Food Facts DuckDB.

    Opened read-only so it can never mutate the artifact built by `setup`
    (and so several instances can share the file across the pipeline and the
    evaluation harness).
    """

    def __init__(self, db_path: Path | None = None, usage: FoodDbUsage | None = None) -> None:
        settings = get_settings()
        path = db_path or settings.off_db
        if not Path(path).exists():
            raise FileNotFoundError(
                f"OFF product DB not found at {path}. Build it first with `just setup` (or `python -m setup`).",
            )
        self._con = duckdb.connect(str(path), read_only=True)
        self._embedding_dim = settings.off_embedding_dim
        self._usage = usage or settings.off_usage
        # Load VSS so the persisted HNSW index is recognised and used to
        # accelerate the semantic ORDER BY; brute-force cosine still works
        # without it, so a missing extension only costs speed, not results.
        try:
            self._con.execute("INSTALL vss")
            self._con.execute("LOAD vss")
        except duckdb.Error as exc:
            log.warning("Could not load DuckDB VSS extension; semantic search will brute-force scan: %s", exc)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> OffFoodDb:
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
            f"  SELECT {_SELECT_COLUMNS}, fts_main_products.match_bm25(code, ?) AS score FROM products"
            f") WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
            [query, limit],
        )

    def _semantic_rows(self, query: str, limit: int) -> list[Row]:
        """Nearest products to `query` by embedding cosine distance.

        Degrades to an empty list (rather than raising) when the DB predates
        the `embedding` column or the embedder is unavailable, so lexical
        search alone still answers the query.
        """
        try:
            vector = embed_query(query)
            return self._rows(
                f"SELECT {_SELECT_COLUMNS} FROM products "  # noqa: S608 (static column list; params are bound)
                f"WHERE embedding IS NOT NULL "
                f"ORDER BY array_cosine_distance(embedding, ?::FLOAT[{self._embedding_dim}]) LIMIT ?",
                [vector, limit],
            )
        except Exception as exc:  # noqa: BLE001 (semantic channel is best-effort)
            log.warning("Semantic search unavailable, using lexical results only: %s", exc)
            return []

    def search(self, query: str, limit: int = 5) -> list[OFFItem]:
        """Return up to `limit` products best matching `query`, ranked by relevance.

        The active channel(s) follow `self._usage`: `only_bm25` / `only_semantic`
        run a single channel, `full` runs both and merges them with reciprocal
        rank fusion (the DuckDB BM25 full-text index for names/brands/categories/
        ingredients, and embedding cosine similarity for paraphrase /
        cross-lingual intent).
        """
        query = query.strip()
        if not query or self._usage is FoodDbUsage.DISABLED:
            return []
        if self._usage is FoodDbUsage.ONLY_BM25:
            rows = self._bm25_rows(query, limit)
        elif self._usage is FoodDbUsage.ONLY_SEMANTIC:
            rows = self._semantic_rows(query, limit)
        else:
            # Over-fetch per channel so fusion has room to reward agreement
            # before the final top-`limit` cut.
            candidates = max(limit * 4, 20)
            bm25 = self._bm25_rows(query, candidates)
            semantic = self._semantic_rows(query, candidates)
            rows = _reciprocal_rank_fusion([bm25, semantic], key=lambda r: r["code"])[:limit]
        return [_row_to_off_item(r) for r in rows]

    def get_food(self, code: str) -> OFFItem:
        """Return the product with `off:<barcode>` `code`, or raise `OFFUnknownFoodCodeError` if absent."""
        rows = self._rows(
            f"SELECT {_SELECT_COLUMNS} FROM products WHERE code = ? LIMIT 1",  # noqa: S608 (static columns)
            [_barcode(code)],
        )
        if not rows:
            raise OFFUnknownFoodCodeError(f"unknown product code: {code!r}")
        return _row_to_off_item(rows[0])
