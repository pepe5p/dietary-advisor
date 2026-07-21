"""Canonical per-100g units and column names for the food DBs.

Both OFF and USDA builds read each source's `_unit` (or `unit_name`) and rescale
amounts into the target unit declared in `TARGET_UNIT`, storing them under
`stored_column(prefix)` (e.g. `sodium_mg_in_100g`). Runtime readers never scale.
A NULL source unit is treated as already-canonical (factor 1.0).
"""

from __future__ import annotations

import duckdb

# Column prefix -> canonical unit stored in `stored_column(prefix)`.
TARGET_UNIT: dict[str, str] = {
    "energy_kcal": "kcal",
    "energy_kj": "kj",
    "proteins": "g",
    "carbohydrates": "g",
    "sugars": "g",
    "fat": "g",
    "saturated_fat": "g",
    "fiber": "g",
    "salt": "g",
    "sodium": "mg",
    "potassium": "mg",
    "calcium": "mg",
    "iron": "mg",
    "vitamin_c": "mg",
    "vitamin_d": "ug",
    "cholesterol": "mg",
}

# Mass units relative to grams. Energy units are identity-only (kcal↔kj are
# separate columns; we never convert between them).
_MASS_BASE: dict[str, float] = {
    "g": 1.0,
    "mg": 1e-3,
    "ug": 1e-6,
    "µg": 1e-6,
}
_ENERGY_UNITS = frozenset({"kcal", "kj"})


def stored_column(prefix: str) -> str:
    """DuckDB column name for a nutrient amount per 100g in `TARGET_UNIT[prefix]`."""
    unit = TARGET_UNIT[prefix]
    if prefix.endswith(f"_{unit}"):
        return f"{prefix}_in_100g"
    return f"{prefix}_{unit}_in_100g"


def unit_ratio(from_unit: str | None, to_unit: str) -> float:
    """Factor `amount * ratio` converts `from_unit` into `to_unit`.

    Unknown or NULL source units return 1.0 (leave the amount as reported).
    """
    if from_unit is None:
        return 1.0
    src = from_unit.strip().casefold()
    dst = to_unit.strip().casefold()
    if not src or src == dst:
        return 1.0
    if src in _ENERGY_UNITS or dst in _ENERGY_UNITS:
        return 1.0
    src_base = _MASS_BASE.get(src)
    dst_base = _MASS_BASE.get(dst)
    if src_base is None or dst_base is None:
        return 1.0
    return src_base / dst_base


def register_unit_ratio_macro(con: duckdb.DuckDBPyConnection) -> None:
    """DuckDB macro mirroring `unit_ratio` for use inside build SELECTs."""
    con.execute(
        """
        CREATE OR REPLACE MACRO unit_ratio(from_unit, to_unit) AS (
            CASE
                WHEN from_unit IS NULL OR trim(CAST(from_unit AS VARCHAR)) = '' THEN 1.0
                WHEN lower(trim(CAST(from_unit AS VARCHAR))) = lower(to_unit) THEN 1.0
                WHEN lower(trim(CAST(from_unit AS VARCHAR))) = 'g'
                     AND lower(to_unit) = 'mg' THEN 1000.0
                WHEN lower(trim(CAST(from_unit AS VARCHAR))) = 'g'
                     AND lower(to_unit) IN ('ug', 'µg') THEN 1000000.0
                WHEN lower(trim(CAST(from_unit AS VARCHAR))) = 'mg'
                     AND lower(to_unit) = 'g' THEN 0.001
                WHEN lower(trim(CAST(from_unit AS VARCHAR))) = 'mg'
                     AND lower(to_unit) IN ('ug', 'µg') THEN 1000.0
                WHEN lower(trim(CAST(from_unit AS VARCHAR))) IN ('ug', 'µg')
                     AND lower(to_unit) = 'g' THEN 0.000001
                WHEN lower(trim(CAST(from_unit AS VARCHAR))) IN ('ug', 'µg')
                     AND lower(to_unit) = 'mg' THEN 0.001
                ELSE 1.0
            END
        )
        """
    )


def scaled_amount_sql(amount_sql: str, unit_sql: str, prefix: str) -> str:
    """SQL expression: `amount` rescaled into `TARGET_UNIT[prefix]`."""
    target = TARGET_UNIT[prefix]
    return f"({amount_sql}) * unit_ratio({unit_sql}, '{target}')"
