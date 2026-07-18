"""Build a local DuckDB database of USDA FoodData Central generic foods.

Source: the FoodData Central CSV exports for Foundation Foods and SR Legacy
(fetched by ``setup.usda_downloading``). These are generic, minimally-processed
/ reference foods - the whole-food counterpart to the branded Polish products in
the OFF DB - so the two databases are searched side by side at runtime.

The relational CSVs (`food`, `food_nutrient`, `nutrient`, `food_category`) are
pivoted into one flat ``foods`` table shaped like the OFF ``products`` table:
one row per food, per-100g nutrient columns paired with their source unit, plus
a full-text index over the description and an embedding for semantic search.
Unlike OFF, USDA reports minerals/vitamins in their canonical units already, so
the runtime reader applies no scaling.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import duckdb

from dietary_advisor.config import Settings
from setup.duckdb_creation import _embed_documents
from setup.usda_downloading import resolve_usda_csv_dirs

log = logging.getLogger(__name__)

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
                "setup/usda_duckdb_creation.py before retrying."
            )


def _has_scientific_name(con: duckdb.DuckDBPyConnection, csv_dirs: list[Path]) -> bool:
    source_sql = _read_csv_sql(_csv_paths(csv_dirs, "food.csv"))
    described = con.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()  # noqa: S608 (paths are local)
    return any(row[0] == "scientific_name" for row in described)


def _wanted_numbers() -> list[float]:
    seen: list[float] = []
    for numbers in _NUTRIENT_NUMBERS.values():
        for n in numbers:
            if n not in seen:
                seen.append(n)
    return seen


def _pivot_columns() -> str:
    """Per-nutrient `<prefix>_100g` amount + `<prefix>_unit` aggregate columns.

    For prefixes backed by several nutrient numbers (energy, sugars) the amount
    and unit are COALESCEd in priority order, so the preferred source wins when
    a food happens to report more than one.
    """
    parts: list[str] = []
    for prefix, numbers in _NUTRIENT_NUMBERS.items():
        amount = " ,\n                ".join(f"max(amount) FILTER (WHERE nbr = {n})" for n in numbers)
        unit = " ,\n                ".join(f"any_value(unit_name) FILTER (WHERE nbr = {n})" for n in numbers)
        if len(numbers) == 1:
            parts.append(f"max(amount) FILTER (WHERE nbr = {numbers[0]}) AS {prefix}_100g")
            parts.append(f"any_value(unit_name) FILTER (WHERE nbr = {numbers[0]}) AS {prefix}_unit")
        else:
            parts.append(f"COALESCE(\n                {amount}\n            ) AS {prefix}_100g")
            parts.append(f"COALESCE(\n                {unit}\n            ) AS {prefix}_unit")
    return ",\n            ".join(parts)


def _build_select_sql(csv_dirs: list[Path], *, has_scientific_name: bool) -> str:
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
    scientific = "scientific_name" if has_scientific_name else "NULL"
    macro_filters = " AND\n          ".join(f"{prefix}_100g IS NOT NULL" for prefix in _REQUIRED_MACROS)

    return f"""
        WITH food AS (
            SELECT
                TRY_CAST(fdc_id AS BIGINT) AS fdc_id,
                data_type,
                description,
                TRY_CAST(food_category_id AS BIGINT) AS food_category_id,
                {scientific} AS scientific_name
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
                food.data_type,
                food.description,
                cat.category,
                food.scientific_name,
                {_pivot_columns()}
            FROM food
            LEFT JOIN cat ON food.food_category_id = cat.id
            LEFT JOIN fnn ON fnn.fdc_id = food.fdc_id
            GROUP BY food.fdc_id, food.data_type, food.description, cat.category, food.scientific_name
        )
        WHERE {macro_filters}
    """  # noqa: S608 (fragments are static config / local paths)


def _document_sql() -> str:
    """SQL expression producing the per-food "what this food is" document.

    USDA generic foods carry no brand or ingredient list, so the searchable
    identity is the description, its food group, and (for Foundation samples)
    the scientific name; `data_type` is spelled out so a query can lean toward
    the freshly-analysed Foundation foods over the frozen SR Legacy ones.
    """
    friendly_type = (
        "CASE data_type "
        "WHEN 'foundation_food' THEN 'Foundation' "
        "WHEN 'sr_legacy_food' THEN 'SR Legacy' "
        "ELSE data_type END"
    )
    parts = (
        "CASE WHEN description IS NOT NULL THEN 'Name: ' || description END",
        "CASE WHEN category IS NOT NULL THEN 'Category: ' || category END",
        "CASE WHEN scientific_name IS NOT NULL THEN 'Scientific name: ' || scientific_name END",
        f"CASE WHEN data_type IS NOT NULL THEN 'Type: ' || ({friendly_type}) END",
    )
    return "concat_ws(chr(10), " + ", ".join(parts) + ")"


def _expected_foods_columns() -> set[str]:
    """The full set of columns a freshly built `foods` table should have."""
    cols = {"fdc_id", "data_type", "description", "category", "scientific_name"}
    for prefix in _NUTRIENT_NUMBERS:
        cols.add(f"{prefix}_100g")
        cols.add(f"{prefix}_unit")
    cols.add("embedding")
    return cols


def _column_comment_sql() -> list[str]:
    statements = [
        "COMMENT ON COLUMN foods.fdc_id IS 'FoodData Central id (primary key). Exposed at runtime as usda:<fdc_id>.'",
        "COMMENT ON COLUMN foods.data_type IS 'FDC data type: foundation_food or sr_legacy_food.'",
    ]
    for prefix in _NUTRIENT_NUMBERS:
        statements.append(
            f"COMMENT ON COLUMN foods.{prefix}_100g IS "
            f"'{prefix} per 100g, in the unit reported by FDC (see {prefix}_unit; already canonical).'"
        )
    return statements


def _embed_foods(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Backfill the `embedding` column and build a VSS HNSW index over it.

    Best-effort HNSW build (the runtime falls back to brute-force cosine, cheap
    at this row count) so a missing/failed VSS extension never blocks the build.
    """
    dim = settings.off_embedding_dim
    con.execute(f"ALTER TABLE foods ADD COLUMN embedding FLOAT[{dim}]")

    rows = con.execute(f"SELECT fdc_id, {_document_sql()} AS document FROM foods").fetchall()  # noqa: S608
    if not rows:
        return
    ids = [r[0] for r in rows]
    documents = [r[1] or "" for r in rows]

    log.info("Embedding %d USDA food documents with %s...", len(documents), settings.off_embedding_model)
    vectors = _embed_documents(documents)

    con.execute(f"CREATE TEMP TABLE _embeddings (fdc_id BIGINT, embedding FLOAT[{dim}])")
    con.executemany("INSERT INTO _embeddings VALUES (?, ?)", list(zip(ids, vectors, strict=True)))
    con.execute(
        "UPDATE foods SET embedding = _embeddings.embedding FROM _embeddings WHERE foods.fdc_id = _embeddings.fdc_id"
    )
    con.execute("DROP TABLE _embeddings")

    try:
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        con.execute("SET hnsw_enable_experimental_persistence = true")
        con.execute("CREATE INDEX foods_embedding_hnsw ON foods USING HNSW (embedding) WITH (metric = 'cosine')")
    except duckdb.Error as exc:
        log.warning("Could not build VSS/HNSW index (falling back to brute-force cosine at query time): %s", exc)


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
    return _expected_foods_columns() - present


def _materialize_usda_db(settings: Settings, csv_dirs: list[Path]) -> Path:
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
            has_scientific = _has_scientific_name(con, csv_dirs)

            log.info("Pivoting USDA Foundation + SR Legacy foods (%d CSV dirs)...", len(csv_dirs))
            con.execute(f"CREATE TABLE foods AS {_build_select_sql(csv_dirs, has_scientific_name=has_scientific)}")
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


def build_usda_db(settings: Settings) -> Path:
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
