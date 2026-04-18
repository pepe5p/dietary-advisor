"""Client for the USDA FoodData Central (FDC) REST API.

USDA FDC is the canonical "ground truth" macronutrient database referenced
throughout the literature review (sec. "Surowe Bazy Danych Makroskładnikowych").
It is the primary backing store for the Symbolic Tool layer.

A local SQLite cache makes the search/get_food calls reproducible and avoids
hammering the API during evaluation runs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

from dietary_advisor.config import get_api_keys, get_settings
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName

log = logging.getLogger(__name__)

FDC_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# Map (USDA nutrient_id, USDA unit) to our canonical (NutrientName, factor).
# Factor converts USDA's reported value into the canonical unit.
# Reference IDs: https://fdc.nal.usda.gov/portal-data/external/dataDictionary
_USDA_NUTRIENT_MAP: dict[int, tuple[NutrientName, float]] = {
    1008: (NutrientName.ENERGY_KCAL, 1.0),       # Energy (kcal)
    2047: (NutrientName.ENERGY_KCAL, 1.0),       # Energy (Atwater general factors)
    2048: (NutrientName.ENERGY_KCAL, 1.0),       # Energy (Atwater specific factors)
    1003: (NutrientName.PROTEIN_G, 1.0),         # Protein (g)
    1005: (NutrientName.CARBS_G, 1.0),           # Carbohydrate, by difference (g)
    1004: (NutrientName.FAT_G, 1.0),             # Total lipid (g)
    1258: (NutrientName.SATURATED_FAT_G, 1.0),   # Fatty acids, total saturated (g)
    1079: (NutrientName.FIBER_G, 1.0),           # Fiber, total dietary (g)
    2000: (NutrientName.SUGAR_G, 1.0),           # Sugars, total
    1093: (NutrientName.SODIUM_MG, 1.0),         # Sodium, Na (mg)
    1092: (NutrientName.POTASSIUM_MG, 1.0),      # Potassium, K (mg)
    1087: (NutrientName.CALCIUM_MG, 1.0),        # Calcium, Ca (mg)
    1089: (NutrientName.IRON_MG, 1.0),           # Iron, Fe (mg)
    1162: (NutrientName.VITAMIN_C_MG, 1.0),      # Vitamin C (mg)
    1114: (NutrientName.VITAMIN_D_UG, 1.0),      # Vitamin D (D2 + D3) (mcg)
    1253: (NutrientName.CHOLESTEROL_MG, 1.0),    # Cholesterol (mg)
}


# Heuristic allergen tagging from USDA description. It is intentionally
# conservative: false positives are far safer than false negatives in
# clinical dietetics.
_ALLERGEN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "milk": ("milk", "cream", "cheese", "butter", "yogurt", "yoghurt", "whey", "casein", "lactose"),
    "eggs": ("egg",),
    "gluten": ("wheat", "barley", "rye", "spelt", "kamut", "triticale", "bulgur", "couscous"),
    "peanuts": ("peanut",),
    "tree_nuts": ("almond", "hazelnut", "walnut", "cashew", "pecan", "pistachio", "macadamia", "brazil nut"),
    "soybeans": ("soy", "soya", "edamame", "tofu", "tempeh"),
    "fish": ("salmon", "tuna", "cod", "trout", "haddock", "sardine", "mackerel", "anchovy"),
    "crustaceans": ("shrimp", "prawn", "crab", "lobster", "crayfish"),
    "molluscs": ("oyster", "mussel", "clam", "scallop", "octopus", "squid"),
    "sesame": ("sesame", "tahini"),
    "celery": ("celery",),
    "mustard": ("mustard",),
    "sulphites": ("sulphite", "sulfite"),
    "lupin": ("lupin", "lupine"),
}

# Vegetarian / vegan classifier hints.
_MEAT_KEYWORDS: tuple[str, ...] = (
    "beef", "pork", "chicken", "turkey", "lamb", "veal", "duck", "goose",
    "bacon", "sausage", "ham", "salami", "prosciutto", "pepperoni",
    "venison", "rabbit", "liver", "kidney",
)
_FISH_KEYWORDS: tuple[str, ...] = sum(
    (_ALLERGEN_KEYWORDS["fish"], _ALLERGEN_KEYWORDS["crustaceans"], _ALLERGEN_KEYWORDS["molluscs"]),
    (),
)


def _derive_tags(description: str) -> list[str]:
    """Build the `tags` list (`contains:<allergen>`, diet-pattern hints)."""
    desc = description.lower()
    tags: list[str] = []
    for allergen, kws in _ALLERGEN_KEYWORDS.items():
        if any(kw in desc for kw in kws):
            tags.append(f"contains:{allergen}")

    has_meat = any(kw in desc for kw in _MEAT_KEYWORDS)
    has_fish = any(kw in desc for kw in _FISH_KEYWORDS)
    has_dairy = any(kw in desc for kw in _ALLERGEN_KEYWORDS["milk"])
    has_eggs = any(kw in desc for kw in _ALLERGEN_KEYWORDS["eggs"])

    if not has_meat and not has_fish:
        tags.append("vegetarian")
        if not has_dairy and not has_eggs:
            tags.append("vegan")
    if has_fish and not has_meat:
        tags.append("pescatarian")
    return tags


def _parse_food_payload(payload: dict[str, Any]) -> FoodItem:
    """Translate a USDA `/food/{fdcId}` payload into our `FoodItem`."""
    description = str(payload.get("description") or payload.get("lowercaseDescription") or "unknown")
    fdc_id = int(payload["fdcId"]) if "fdcId" in payload else None

    nutrients: dict[NutrientName, float] = {}
    for fn in payload.get("foodNutrients", []) or []:
        # FDC has multiple shapes depending on data type. Handle both.
        nutrient_block = fn.get("nutrient") or {}
        nutrient_id = nutrient_block.get("id") or fn.get("nutrientId")
        amount = fn.get("amount")
        if amount is None:
            amount = fn.get("value")
        if nutrient_id is None or amount is None:
            continue
        try:
            nid = int(nutrient_id)
        except (TypeError, ValueError):
            continue
        mapping = _USDA_NUTRIENT_MAP.get(nid)
        if mapping is None:
            continue
        canonical, factor = mapping
        # If we already have an entry (e.g. multiple energy entries), keep the first.
        nutrients.setdefault(canonical, float(amount) * factor)

    return FoodItem(
        fdc_id=fdc_id,
        name=description,
        description=description,
        nutrients_per_100g=nutrients,
        tags=_derive_tags(description),
    )


class _SqliteCache:
    """Bare-bones key-value cache with TTL."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, ts REAL NOT NULL)",
        )
        self._conn.commit()

    def get(self, key: str, ttl_s: float | None = None) -> Any | None:
        row = self._conn.execute("SELECT value, ts FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        value, ts = row
        if ttl_s is not None and (time.time() - ts) > ttl_s:
            return None
        return json.loads(value)

    def set(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class USDAClient:
    """Thin httpx-based client for USDA FoodData Central with local caching."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: Path | None = None,
        timeout_s: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        keys = get_api_keys()
        self._api_key = api_key or keys.usda_api_key
        self._cache = _SqliteCache(cache_path or settings.usda_cache)
        self._client = client or httpx.Client(
            base_url=FDC_BASE_URL,
            timeout=timeout_s or settings.request_timeout_s,
        )
        self._owned_client = client is None

    def close(self) -> None:
        if self._owned_client:
            self._client.close()
        self._cache.close()

    def __enter__(self) -> USDAClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                 json_body: dict[str, Any] | None = None, retries: int = 3) -> dict[str, Any]:
        params = dict(params or {})
        params["api_key"] = self._api_key
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = self._client.request(method, path, params=params, json=json_body)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = exc
                wait = min(2 ** (attempt - 1), 8)
                log.warning("USDA request failed (attempt %d/%d): %s. Backing off %ds.",
                            attempt, retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"USDA request {method} {path} failed after {retries} attempts: {last_exc}")

    def search(self, query: str, page_size: int = 5) -> list[FoodItem]:
        """Search the FDC database, return up to `page_size` `FoodItem`s."""
        cache_key = f"search::{page_size}::{query.strip().lower()}"
        cached = self._cache.get(cache_key, ttl_s=7 * 86400)
        if cached is not None:
            payload = cached
        else:
            payload = self._request(
                "POST",
                "/foods/search",
                json_body={"query": query, "pageSize": page_size, "dataType": ["Foundation", "SR Legacy"]},
            )
            self._cache.set(cache_key, payload)
        foods = payload.get("foods", []) or []
        return [_parse_food_payload(f) for f in foods]

    def get_food(self, fdc_id: int) -> FoodItem:
        """Fetch a single FDC entry by ID."""
        cache_key = f"food::{fdc_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return _parse_food_payload(cached)
        payload = self._request("GET", f"/food/{fdc_id}")
        self._cache.set(cache_key, payload)
        return _parse_food_payload(payload)

    def search_first(self, queries: Iterable[str]) -> list[FoodItem]:
        """Run multiple searches in sequence, return first match for each."""
        out: list[FoodItem] = []
        for q in queries:
            results = self.search(q, page_size=1)
            if results:
                out.append(results[0])
        return out
