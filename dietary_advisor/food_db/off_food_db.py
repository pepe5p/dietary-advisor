"""Local Open Food Facts product database (DuckDB), the food source of truth.

Replaces the former USDA REST client: instead of hitting a remote API, we
query the local `.data/off/off_pl.duckdb` built by the `setup` command. There is
deliberately no lookup abstraction - this one concrete class is the only food
source in the system.

The agent searches by product name (`search`) and gets the best-matching
records; evaluation resolves a plan's barcodes back to their canonical
nutrients (`get_food`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb

from dietary_advisor.config import FoodDbUsage, get_settings
from dietary_advisor.food_db.embeddings import embed_query
from dietary_advisor.food_db.fusion import reciprocal_rank_fusion as _reciprocal_rank_fusion
from dietary_advisor.food_db.models import OFFItem
from dietary_advisor.schemas.nutrition import NutrientName

log = logging.getLogger(__name__)

# OFF per-100g column -> (canonical nutrient, factor to canonical unit).
# OFF normalizes `*_100g` to grams (energy in kcal); the Totaller expects the
# canonical units declared in schemas/nutrition.py, so minerals/vitamins in
# grams are scaled to mg / ug here.
_NUTRIENT_FACTORS: tuple[tuple[str, NutrientName, float], ...] = (
    ("energy_kcal_100g", NutrientName.ENERGY_KCAL, 1.0),
    ("proteins_100g", NutrientName.PROTEIN_G, 1.0),
    ("carbohydrates_100g", NutrientName.CARBS_G, 1.0),
    ("fat_100g", NutrientName.FAT_G, 1.0),
    ("saturated_fat_100g", NutrientName.SATURATED_FAT_G, 1.0),
    ("fiber_100g", NutrientName.FIBER_G, 1.0),
    ("sugars_100g", NutrientName.SUGAR_G, 1.0),
    ("sodium_100g", NutrientName.SODIUM_MG, 1000.0),
    ("potassium_100g", NutrientName.POTASSIUM_MG, 1000.0),
    ("calcium_100g", NutrientName.CALCIUM_MG, 1000.0),
    ("iron_100g", NutrientName.IRON_MG, 1000.0),
    ("vitamin_c_100g", NutrientName.VITAMIN_C_MG, 1000.0),
    ("vitamin_d_100g", NutrientName.VITAMIN_D_UG, 1_000_000.0),
    ("cholesterol_100g", NutrientName.CHOLESTEROL_MG, 1000.0),
)

# OFF allergen slug (without the `en:` language prefix) -> our Allergen value,
# matching the `contains:<allergen>` tags the rule validator checks.
_OFF_ALLERGEN_TO_TAG: dict[str, str] = {
    "milk": "milk",
    "gluten": "gluten",
    "cereals-with-gluten": "gluten",
    "soybeans": "soybeans",
    "eggs": "eggs",
    "mustard": "mustard",
    "nuts": "tree nuts",
    "celery": "celery",
    "peanuts": "peanuts",
    "fish": "fish",
    "sesame-seeds": "sesame",
    "sesame": "sesame",
    "sulphur-dioxide-and-sulphites": "sulphites",
    "crustaceans": "crustaceans",
    "lupin": "lupin",
    "molluscs": "molluscs",
}

_SELECT_COLUMNS = (
    "code, product_name, product_name_pl, generic_name, "
    "labels_tags, allergens_tags, traces_tags, "
    "brands, brands_tags, categories, categories_tags, compared_to_category, ingredients_text, "
    + ", ".join(col for col, _, _ in _NUTRIENT_FACTORS)
)


def _diet_tags(labels: list[str]) -> list[str]:
    tags: list[str] = []
    low = [t.lower() for t in labels]
    is_vegan = any("vegan" in t and "non-vegan" not in t and "no-vegan" not in t for t in low)
    is_vegetarian = is_vegan or any("vegetarian" in t and "non-vegetarian" not in t for t in low)
    if is_vegan:
        tags.append("vegan")
    if is_vegetarian:
        tags.append("vegetarian")
    return tags


def _allergen_tags(*tag_lists: list[str] | None) -> list[str]:
    out: list[str] = []
    for tags in tag_lists:
        for raw in tags or []:
            slug = raw.split(":", 1)[-1].lower() if ":" in raw else raw.lower()
            mapped = _OFF_ALLERGEN_TO_TAG.get(slug)
            if mapped:
                tag = f"contains:{mapped}"
                if tag not in out:
                    out.append(tag)
    return out


def _row_to_off_item(row: Mapping[str, Any]) -> OFFItem:
    """Map a `products` row to an `OFFItem` (pure; no DB access)."""
    nutrients: dict[NutrientName, float] = {}
    for col, nutrient, factor in _NUTRIENT_FACTORS:
        value = row.get(col)
        if value is None:
            continue
        scaled = float(value) * factor
        if scaled >= 0:
            nutrients[nutrient] = scaled

    name = row.get("product_name") or row.get("product_name_pl") or row.get("code") or "unknown"
    tags = _diet_tags(row.get("labels_tags") or []) + _allergen_tags(
        row.get("allergens_tags"),
        row.get("traces_tags"),
    )
    return OFFItem(
        code=row["code"],
        name=str(name),
        description=row.get("generic_name"),
        nutrients_per_100g=nutrients,
        tags=tags,
        brands=row.get("brands"),
        brands_tags=row.get("brands_tags") or [],
        categories=row.get("categories"),
        categories_tags=row.get("categories_tags") or [],
        compared_to_category=row.get("compared_to_category"),
        ingredients_text=row.get("ingredients_text"),
    )


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

    def _rows(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cur = self._con.execute(sql, params)
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def _bm25_rows(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Lexical (BM25) candidates from the full-text index built by `setup`."""
        return self._rows(
            f"SELECT * FROM ("  # noqa: S608 (static column list; params are bound)
            f"  SELECT {_SELECT_COLUMNS}, fts_main_products.match_bm25(code, ?) AS score FROM products"
            f") WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
            [query, limit],
        )

    def _semantic_rows(self, query: str, limit: int) -> list[dict[str, Any]]:
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
        """Return the product with barcode `code`, or raise `KeyError` if absent."""
        rows = self._rows(
            f"SELECT {_SELECT_COLUMNS} FROM products WHERE code = ? LIMIT 1",  # noqa: S608 (static columns)
            [code],
        )
        if not rows:
            raise KeyError(f"unknown product code: {code!r}")
        food = _row_to_off_item(rows[0])
        return food
