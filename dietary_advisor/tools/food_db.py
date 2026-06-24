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

from dietary_advisor.config import get_settings
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName

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
    "labels_tags, allergens_tags, traces_tags, " + ", ".join(col for col, _, _ in _NUTRIENT_FACTORS)
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


def _row_to_food_item(row: Mapping[str, Any]) -> FoodItem:
    """Map a `products` row to a canonical `FoodItem` (pure; no DB access)."""
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
    return FoodItem(
        from_open_food_facts=True,
        code=row.get("code"),
        name=str(name),
        description=row.get("generic_name"),
        nutrients_per_100g=nutrients,
        tags=tags,
    )


class OffFoodDb:
    """Read-only accessor over the local Open Food Facts DuckDB.

    Opened read-only so it can never mutate the artifact built by `setup`
    (and so several instances can share the file across the pipeline and the
    evaluation harness).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or get_settings().off_db
        if not Path(path).exists():
            raise FileNotFoundError(
                f"OFF product DB not found at {path}. Build it first with `just setup` (or `python -m setup`).",
            )
        self._con = duckdb.connect(str(path), read_only=True)

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

    def search(self, query: str, limit: int = 5) -> list[FoodItem]:
        """Return up to `limit` products best matching `query`, ranked by relevance.

        Uses the DuckDB full-text (BM25) index built by `setup`; when the query
        yields no full-text hits (e.g. a very short or unusual term) it falls
        back to a substring match on the product name.
        """
        query = query.strip()
        if not query:
            return []
        ranked = self._rows(
            f"SELECT * FROM ("  # noqa: S608 (static column list; params are bound)
            f"  SELECT {_SELECT_COLUMNS}, fts_main_products.match_bm25(code, ?) AS score FROM products"
            f") WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
            [query, limit],
        )
        if ranked:
            results = [_row_to_food_item(r) for r in ranked]
        else:
            like = self._rows(
                f"SELECT {_SELECT_COLUMNS} FROM products "  # noqa: S608 (static column list; params are bound)
                f"WHERE product_name ILIKE '%' || ? || '%' OR product_name_pl ILIKE '%' || ? || '%' LIMIT ?",
                [query, query, limit],
            )
            results = [_row_to_food_item(r) for r in like]
        log.info("food_db.search(%r, limit=%d) -> %d result(s)", query, limit, len(results))
        return results

    def get_food(self, code: str) -> FoodItem:
        """Return the product with barcode `code`, or raise `KeyError` if absent."""
        rows = self._rows(
            f"SELECT {_SELECT_COLUMNS} FROM products WHERE code = ? LIMIT 1",  # noqa: S608 (static columns)
            [code],
        )
        if not rows:
            raise KeyError(f"unknown product code: {code!r}")
        food = _row_to_food_item(rows[0])
        log.info("food_db.get_food(%r) -> %r", code, food.name)
        return food
