"""Build a local DuckDB database of USDA FoodData Central generic foods.

Source: the FoodData Central CSV exports for Foundation Foods and SR Legacy
(fetched by ``setup.usda.downloading``). These are generic, minimally-processed
/ reference foods - the whole-food counterpart to the branded Polish products in
the OFF DB - so the two databases are searched side by side at runtime.

The relational CSVs (`food`, `food_nutrient`, `nutrient`, `food_category`) are
pivoted into one flat ``foods`` table shaped like the OFF ``products`` table:
one row per food, per-100g nutrient columns in the canonical units from
`setup.units.TARGET_UNIT`, plus a full-text index over the description and an
embedding for semantic search. USDA usually already reports canonical units;
the build still reads each `unit_name` and rescales so both DBs share one
convention.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import duckdb

from setup.embedding import embed_table
from setup.settings import SetupSettings
from setup.units import register_unit_ratio_macro, scaled_amount_sql, stored_column, TARGET_UNIT
from setup.usda.downloading import resolve_usda_csv_dirs

log = logging.getLogger(__name__)

# Columns streamed into the Python document builder for semantic embeddings.
_DOCUMENT_COLUMNS: tuple[str, ...] = ("description", "category")

# Canonical nutrient column prefix -> the FDC `nutrient_nbr` code(s) that supply
# it, in priority order. `nutrient_nbr` (203, 208, ...) is the stable, human-
# readable nutrient identifier shared across both datasets. Energy prefers the
# directly-reported kcal (208) but falls back to the Atwater general/specific
# factors (957/958) that some Foundation foods carry instead, so the macro-
# completeness filter below doesn't needlessly drop them. Column prefixes match
# the OFF build so the runtime readers share one per-100g column convention.
_NUTRIENT_NUMBERS: dict[str, tuple[float, ...]] = {
    "energy_kcal": (208.0, 957.0, 958.0),
    "proteins": (203.0,),
    "carbohydrates": (205.0,),
    "fat": (204.0,),
    "saturated_fat": (606.0,),
    "fiber": (291.0,),
    "sugars": (269.0, 269.3),
    "sodium": (307.0,),
    "potassium": (306.0,),
    "calcium": (301.0,),
    "iron": (303.0,),
    "vitamin_c": (401.0,),
    "vitamin_d": (328.0,),
    "cholesterol": (601.0,),
}

# Macros that must be present (non-null) for a food to be usable by the
# Totaller; mirrors the OFF completeness filter.
_REQUIRED_MACROS: tuple[str, ...] = ("energy_kcal", "proteins", "carbohydrates", "fat")

# Only the aggregated food rows are wanted; each archive's `food.csv` also
# carries sample/acquisition rows (other data_types) that are lab provenance.
_FOOD_DATA_TYPES: tuple[str, ...] = ("foundation_food", "sr_legacy_food")

# Per-CSV columns the build query reads. Verified with a live DESCRIBE so a
# future FDC schema change fails loudly instead of silently NULLing data.
_REQUIRED_CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "food.csv": ("fdc_id", "data_type", "description", "food_category_id"),
    "food_nutrient.csv": ("fdc_id", "nutrient_id", "amount"),
    "nutrient.csv": ("id", "nutrient_nbr", "unit_name"),
    "food_category.csv": ("id", "description"),
}


class UsdaSchemaError(RuntimeError):
    """Raised when a FoodData Central CSV export lacks the columns we depend on."""


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _read_csv_sql(paths: list[Path]) -> str:
    """A `read_csv(...)` call over `paths`, everything as VARCHAR.

    Reading `all_varchar` sidesteps type-sniffing failures on the messy footnote
    columns, and `union_by_name` tolerates the schema drift between the 2024
    Foundation and 2018 SR Legacy vintages (missing columns become NULL).
    """
    listed = ", ".join(f"'{_sql_quote(str(p))}'" for p in paths)
    return f"read_csv([{listed}], header = true, all_varchar = true, union_by_name = true)"


def _csv_paths(csv_dirs: list[Path], name: str) -> list[Path]:
    return [d / name for d in csv_dirs]


def _check_schema(con: duckdb.DuckDBPyConnection, csv_dirs: list[Path]) -> None:
    """Verify every source CSV still exposes the columns the build reads."""
    for name, required in _REQUIRED_CSV_COLUMNS.items():
        source_sql = _read_csv_sql(_csv_paths(csv_dirs, name))
        described = con.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()  # noqa: S608 (paths are local)
        available = {row[0] for row in described}
        missing = [c for c in required if c not in available]
        if missing:
            raise UsdaSchemaError(
                f"USDA {name} export is missing expected column(s): {missing}. "
                "The FoodData Central schema likely changed - update "
                "setup/usda/duckdb_creation.py before retrying."
            )


def _wanted_numbers() -> list[float]:
    seen: list[float] = []
    for numbers in _NUTRIENT_NUMBERS.values():
        for n in numbers:
            if n not in seen:
                seen.append(n)
    return seen


def _pivot_columns() -> str:
    """Per-nutrient columns named by `stored_column(prefix)` in `TARGET_UNIT[prefix]`.

    For prefixes backed by several nutrient numbers (energy, sugars) the amount
    and unit are COALESCEd in priority order, so the preferred source wins when
    a food happens to report more than one; then `unit_ratio` rescales.
    """
    parts: list[str] = []
    for prefix, numbers in _NUTRIENT_NUMBERS.items():
        if len(numbers) == 1:
            amount = f"max(amount) FILTER (WHERE nbr = {numbers[0]})"
            unit = f"any_value(unit_name) FILTER (WHERE nbr = {numbers[0]})"
        else:
            amount_parts = " ,\n                ".join(f"max(amount) FILTER (WHERE nbr = {n})" for n in numbers)
            unit_parts = " ,\n                ".join(f"any_value(unit_name) FILTER (WHERE nbr = {n})" for n in numbers)
            amount = f"COALESCE(\n                {amount_parts}\n            )"
            unit = f"COALESCE(\n                {unit_parts}\n            )"
        parts.append(f"{scaled_amount_sql(amount, unit, prefix)} AS {stored_column(prefix)}")
    return ",\n            ".join(parts)


def _build_select_sql(csv_dirs: list[Path]) -> str:
    """The CREATE TABLE ... AS SELECT body pivoting the FDC CSVs into `foods`.

    All interpolated fragments come from static config or the local CSV paths,
    not from untrusted external input.
    """
    food_sql = _read_csv_sql(_csv_paths(csv_dirs, "food.csv"))
    fn_sql = _read_csv_sql(_csv_paths(csv_dirs, "food_nutrient.csv"))
    nutrient_sql = _read_csv_sql(_csv_paths(csv_dirs, "nutrient.csv"))
    category_sql = _read_csv_sql(_csv_paths(csv_dirs, "food_category.csv"))

    data_types = ", ".join(f"'{dt}'" for dt in _FOOD_DATA_TYPES)
    wanted = ", ".join(str(n) for n in _wanted_numbers())
    macro_filters = " AND\n          ".join(f"{stored_column(prefix)} IS NOT NULL" for prefix in _REQUIRED_MACROS)

    return f"""
        WITH food AS (
            SELECT
                TRY_CAST(fdc_id AS BIGINT) AS fdc_id,
                description,
                TRY_CAST(food_category_id AS BIGINT) AS food_category_id
            FROM {food_sql}
            WHERE data_type IN ({data_types})
              AND TRY_CAST(fdc_id AS BIGINT) IS NOT NULL
              AND description IS NOT NULL
        ),
        cat AS (
            SELECT TRY_CAST(id AS BIGINT) AS id, any_value(description) AS category
            FROM {category_sql}
            GROUP BY 1
        ),
        nut AS (
            SELECT
                TRY_CAST(id AS BIGINT) AS id,
                any_value(TRY_CAST(nutrient_nbr AS DOUBLE)) AS nbr,
                any_value(unit_name) AS unit_name
            FROM {nutrient_sql}
            GROUP BY 1
        ),
        fnn AS (
            SELECT
                TRY_CAST(fn.fdc_id AS BIGINT) AS fdc_id,
                nut.nbr AS nbr,
                nut.unit_name AS unit_name,
                TRY_CAST(fn.amount AS DOUBLE) AS amount
            FROM {fn_sql} AS fn
            JOIN nut ON TRY_CAST(fn.nutrient_id AS BIGINT) = nut.id
            WHERE nut.nbr IN ({wanted})
        )
        SELECT * FROM (
            SELECT
                food.fdc_id,
                food.description,
                cat.category,
                {_pivot_columns()}
            FROM food
            LEFT JOIN cat ON food.food_category_id = cat.id
            LEFT JOIN fnn ON fnn.fdc_id = food.fdc_id
            GROUP BY food.fdc_id, food.description, cat.category
        )
        WHERE {macro_filters}
    """  # noqa: S608 (fragments are static config / local paths)


def _indefinite_article(phrase: str) -> str:
    return "An" if phrase[:1].casefold() in "aeiou" else "A"


def _build_document(row: dict[str, Any]) -> str:
    """Per-food prose for the embedding model.

    Prose encodes role relationships better than labelled lines. USDA generic
    foods carry no brand, so identity is the description and food group.
    Both are always non-null/non-empty in the built table.
    """
    description = str(row["description"])
    category = str(row["category"])
    return f"{description}. {_indefinite_article(category)} {category} product."


def _expected_foods_columns() -> set[str]:
    """The full set of columns a freshly built `foods` table should have."""
    cols = {"fdc_id", "description", "category"}
    for prefix in _NUTRIENT_NUMBERS:
        cols.add(stored_column(prefix))
    cols.add("embedding")
    return cols


def _column_comment_sql() -> list[str]:
    statements = [
        "COMMENT ON COLUMN foods.fdc_id IS 'FoodData Central id (primary key). Exposed at runtime as usda:<fdc_id>.'",
    ]
    for prefix in _NUTRIENT_NUMBERS:
        unit = TARGET_UNIT[prefix]
        col = stored_column(prefix)
        statements.append(f"COMMENT ON COLUMN foods.{col} IS '{prefix} per 100g, in {unit} (canonical).'")
    return statements


def _embed_foods(con: duckdb.DuckDBPyConnection, settings: SetupSettings) -> None:
    """Backfill the `embedding` column and build a VSS HNSW index over it."""
    embed_table(
        con,
        table="foods",
        id_column="fdc_id",
        id_type="BIGINT",
        columns=_DOCUMENT_COLUMNS,
        build_document=_build_document,
        dim=settings.off_embedding_dim,
        model_name=settings.off_embedding_model,
    )


def _stale_foods_columns(target: Path) -> set[str] | None:
    """Expected `foods` columns missing from the DB at `target`.

    Returns `None` when there is no usable DB to reuse (absent/empty/unreadable)
    and an empty set when the existing DB already has every expected column.
    """
    if not target.exists() or target.stat().st_size == 0:
        return None
    try:
        con = duckdb.connect(str(target), read_only=True)
    except duckdb.Error:
        return None
    try:
        described = con.execute("DESCRIBE foods").fetchall()
    except duckdb.Error:
        return None
    finally:
        con.close()
    present = {row[0] for row in described}
    missing = _expected_foods_columns() - present
    stale_units = {c for c in present if c.endswith("_unit")}
    return missing | stale_units


def _materialize_usda_db(settings: SetupSettings, csv_dirs: list[Path]) -> Path:
    """Pivot the FDC CSVs into a fresh `foods` DuckDB at `settings.usda_db`.

    Writes to a sibling temp file and renames on success so an interrupted build
    never leaves a half-written DB in place.
    """
    target = settings.usda_db
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f".tmp-{os.getpid()}-{int(time.time())}{target.suffix}")

    try:
        con = duckdb.connect(str(tmp_path))
        try:
            _check_schema(con, csv_dirs)
            register_unit_ratio_macro(con)

            log.info("Pivoting USDA Foundation + SR Legacy foods (%d CSV dirs)...", len(csv_dirs))
            con.execute(f"CREATE TABLE foods AS {_build_select_sql(csv_dirs)}")
            row_count = con.execute("SELECT count(*) FROM foods").fetchone()[0]  # type: ignore[index]

            for stmt in _column_comment_sql():
                con.execute(stmt)

            try:
                con.execute("PRAGMA create_fts_index('foods', 'fdc_id', 'description', overwrite=1)")
            except duckdb.Error as exc:
                log.warning("Could not build FTS index (falling back to ILIKE at query time): %s", exc)

            _embed_foods(con, settings)

            con.execute(
                "CREATE TABLE usda_meta "
                "(built_at TIMESTAMP, foundation_url VARCHAR, sr_legacy_url VARCHAR, row_count BIGINT, "
                "embedding_model VARCHAR)"
            )
            con.execute(
                "INSERT INTO usda_meta VALUES (now(), ?, ?, ?, ?)",
                [
                    settings.usda_foundation_source_url,
                    settings.usda_sr_legacy_source_url,
                    row_count,
                    settings.off_embedding_model,
                ],
            )
        finally:
            con.close()
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    os.replace(tmp_path, target)
    log.info("Built USDA food DB at %s (%d foods with complete macros).", target, row_count)
    return target


def build_usda_db(settings: SetupSettings) -> Path:
    """Ensure the local USDA food DB exists and is current at `settings.usda_db`.

    Skips the build only when an existing DB already has every expected column;
    a DB predating a schema change is rebuilt. Otherwise the Foundation + SR
    Legacy CSV archives are resolved to the local cache and pivoted into `foods`.
    """
    target = settings.usda_db
    missing = _stale_foods_columns(target)
    if missing == set():
        log.info("USDA food DB already present and current at %s, skipping build.", target)
        return target
    if missing:
        log.info("USDA food DB at %s is missing expected column(s) %s; rebuilding.", target, sorted(missing))

    csv_dirs = resolve_usda_csv_dirs(settings)
    return _materialize_usda_db(settings, csv_dirs)
