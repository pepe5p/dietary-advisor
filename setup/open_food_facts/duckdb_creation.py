"""Build a local DuckDB database of Polish Open Food Facts products.

Source: the Open Food Facts Hugging Face Parquet export (already slimmed vs.
the raw MongoDB/JSONL dump), fetched to a local cache by
``setup.open_food_facts.downloading``. We filter down to products sold in
Poland with a complete macro profile, drop unidentifiable rows (see
``_trash_predicate``), keep only the columns useful to the dietary advisor,
and materialize them into a single portable ``.duckdb`` file plus a full-text
index over product names/brands/categories (see ``_FTS_FIELDS``) for lookup
by name and descriptive text.

Nutrient amounts are stored per 100g in the canonical units declared in
`setup.units.TARGET_UNIT`, converted at build time from each source `_unit`.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import duckdb

from setup.embedding import embed_table
from setup.open_food_facts.downloading import resolve_source_parquet
from setup.settings import SetupSettings
from setup.units import register_unit_ratio_macro, scaled_amount_sql, stored_column, TARGET_UNIT

log = logging.getLogger(__name__)

# Columns fed to the BM25 full-text index. Beyond names/brands this includes
# `categories` so lexical search can hit the English descriptive text of a
# product whose name is only in another language (e.g. the Polish
# "Filet z kaczki" is reachable via its "duck" category). Single source of
# truth: the build creates the index from this, and `build_off_db` compares it
# against an existing DB's indexed fields to decide whether an in-place
# FTS-only rebuild is needed. `categories_tags` is left out deliberately - it's
# a VARCHAR[] the FTS indexer rejects, and its content duplicates `categories`.
_FTS_FIELDS: tuple[str, ...] = (
    "product_name",
    "product_name_pl",
    "brands",
    "categories",
)

# Columns that identify *what a food is* and can form an embedding document
# (name or category). A row with all of these empty cannot be matched by search
# even if it still carries a brand, ingredients, or label/allergen tags
# (e.g. barcode 5900766003084: brand "Polskie młyny" but no name or category -
# useless to the agent), so setup drops it. Brand/ingredients/tags are
# deliberately excluded: brand never says what the product is, and ingredients/
# tag-only rows cannot build a non-empty embedding document.
_IDENTIFYING_TEXT_COLUMNS: tuple[str, ...] = (
    "product_name",
    "product_name_pl",
    "categories",
    "compared_to_category",
)

# Columns streamed into the Python document builder for semantic embeddings.
_DOCUMENT_COLUMNS: tuple[str, ...] = (
    "product_name",
    "product_name_pl",
    "categories",
    "compared_to_category",
    "brands",
)

_TAG_PREFIX_RE = re.compile(r"^[a-z]{2,3}:")


def _trash_predicate() -> str:
    """SQL boolean, true for rows with no identifying content (see `_IDENTIFYING_TEXT_COLUMNS`).

    Shared by the fresh-build filter and the in-place cleanup so both agree on
    exactly which rows are worthless.
    """
    text = [f"NULLIF(trim({c}), '') IS NULL" for c in _IDENTIFYING_TEXT_COLUMNS]
    return "(" + " AND ".join(text) + ")"


# Atwater estimate of kcal/100g from the macronutrients, used to detect energy
# values that can't be reconciled with the product's own macros. Label kcal may
# legitimately include fiber at 2 kcal/g, so the mismatch check accepts anything
# between the pure 9/4/4 estimate and the fiber-credited one. Rounding on
# low-energy labels (teas, waters: 0.1 kcal vs 0.2 expected) makes a purely
# relative margin delete half of all sub-50-kcal products, hence the absolute
# floor below.
_ENERGY_KCAL_COL = stored_column("energy_kcal")
_FAT_COL = stored_column("fat")
_PROTEINS_COL = stored_column("proteins")
_CARBS_COL = stored_column("carbohydrates")
_FIBER_COL = stored_column("fiber")
_ATWATER_KCAL = f"({_FAT_COL} * 9 + {_PROTEINS_COL} * 4 + {_CARBS_COL} * 4)"
_ATWATER_KCAL_WITH_FIBER = f"({_ATWATER_KCAL} + coalesce({_FIBER_COL}, 0) * 2)"
_ENERGY_MARGIN = 0.05
_ENERGY_MARGIN_FLOOR_KCAL = 20

# Decimal-shift factors tried (in order, first match wins) when repairing an
# energy value: a mis-placed decimal point or a per-serving/per-100g mix-up
# shows up as kcal off by a power of ten from the Atwater estimate.
_ENERGY_TYPO_FACTORS: tuple[float, ...] = (10, 100, 1000, 0.1, 0.01)

# Pure fat is 9 kcal/g, so no real food exceeds 900 kcal/100g; anything above is
# a data error (kJ mislabelled as kcal, per-serving values, etc.).
_MAX_KCAL_100G = 900


def _energy_checkable() -> str:
    """SQL boolean: the four columns the energy checks need are all present."""
    cols = (_ENERGY_KCAL_COL, _FAT_COL, _PROTEINS_COL, _CARBS_COL)
    return "(" + " AND ".join(f"{c} IS NOT NULL" for c in cols) + ")"


def _energy_typo_predicate(factor: float) -> str:
    """SQL boolean: dividing kcal by `factor` reconciles it with the Atwater estimate."""
    return (
        f"{_energy_checkable()} AND {_ATWATER_KCAL} > 0 "
        f"AND abs({_ENERGY_KCAL_COL} / {factor} - {_ATWATER_KCAL}) <= {_ENERGY_MARGIN} * {_ATWATER_KCAL}"
    )


def _energy_mismatch_predicate() -> str:
    """SQL boolean: kcal falls outside the plausible band derived from the macros.

    The band spans the pure Atwater estimate to the fiber-credited one, widened
    on each side by the 5% margin with a `_ENERGY_MARGIN_FLOOR_KCAL` floor.
    """
    margin_low = f"greatest({_ENERGY_MARGIN} * {_ATWATER_KCAL}, {_ENERGY_MARGIN_FLOOR_KCAL})"
    margin_high = f"greatest({_ENERGY_MARGIN} * {_ATWATER_KCAL_WITH_FIBER}, {_ENERGY_MARGIN_FLOOR_KCAL})"
    return (
        f"{_energy_checkable()} AND ("
        f"{_ENERGY_KCAL_COL} < {_ATWATER_KCAL} - {margin_low}"
        f" OR {_ENERGY_KCAL_COL} > {_ATWATER_KCAL_WITH_FIBER} + {margin_high})"
    )


def _clean_energy(con: duckdb.DuckDBPyConnection) -> int:
    """Repair or drop products whose energy disagrees with their macros.

    Ordered so repairs run before deletions: (1) rescale kcal off by a power of
    ten from the Atwater estimate, (2) delete impossible values above
    `_MAX_KCAL_100G`, (3) delete whatever still can't be reconciled. Returns the
    number of rows deleted (steps 2+3), which the caller uses to decide whether
    the FTS corpus needs rebuilding.
    """
    for factor in _ENERGY_TYPO_FACTORS:
        pred = _energy_typo_predicate(factor)
        fixed = con.execute(f"SELECT count(*) FROM products WHERE {pred}").fetchone()[0]  # type: ignore[index]  # noqa: S608
        if fixed:
            con.execute(
                f"UPDATE products SET {_ENERGY_KCAL_COL} = {_ENERGY_KCAL_COL} / {factor} WHERE {pred}"  # noqa: S608
            )
            log.info("Rescaled %s by 1/%s for %d product(s).", _ENERGY_KCAL_COL, factor, fixed)

    over = con.execute(
        f"SELECT count(*) FROM products WHERE {_ENERGY_KCAL_COL} > {_MAX_KCAL_100G}"  # noqa: S608
    ).fetchone()[0]  # type: ignore[index]
    if over:
        con.execute(f"DELETE FROM products WHERE {_ENERGY_KCAL_COL} > {_MAX_KCAL_100G}")  # noqa: S608
        log.info("Removed %d product(s) with %s > %d.", over, _ENERGY_KCAL_COL, _MAX_KCAL_100G)

    mismatch = _energy_mismatch_predicate()
    bad = con.execute(f"SELECT count(*) FROM products WHERE {mismatch}").fetchone()[0]  # type: ignore[index]  # noqa: S608
    if bad:
        con.execute(f"DELETE FROM products WHERE {mismatch}")  # noqa: S608
        log.info("Removed %d product(s) whose energy disagrees with their macros.", bad)

    return over + bad


# OFF nutrient key (as found in the `nutriments` struct list) -> our column
# prefix. Values are stored per-100g in `TARGET_UNIT[prefix]` after conversion.
_NUTRIENT_COLUMNS: dict[str, str] = {
    "energy-kcal": "energy_kcal",
    "energy-kj": "energy_kj",
    "proteins": "proteins",
    "carbohydrates": "carbohydrates",
    "sugars": "sugars",
    "fat": "fat",
    "saturated-fat": "saturated_fat",
    "fiber": "fiber",
    "salt": "salt",
    "sodium": "sodium",
    "potassium": "potassium",
    "calcium": "calcium",
    "iron": "iron",
    "vitamin-c": "vitamin_c",
    "vitamin-d": "vitamin_d",
    "cholesterol": "cholesterol",
}

# The macro nutrients that must be present (non-null) for a product to be
# considered usable by the Totaller; this is our completeness filter.
_REQUIRED_MACROS: tuple[str, ...] = ("energy-kcal", "proteins", "carbohydrates", "fat")

# Source columns the build query depends on. Verified against a live
# `DESCRIBE` at build time so a future OFF export schema change fails loudly
# (with an actionable message) instead of silently dropping/mis-mapping data.
_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "code",
    "product_name",
    "ingredients_text",
    "brands",
    "brands_tags",
    "quantity",
    "serving_size",
    "serving_quantity",
    "product_quantity",
    "product_quantity_unit",
    "nutrition_data_per",
    "categories",
    "categories_tags",
    "compared_to_category",
    "labels_tags",
    "allergens_tags",
    "traces_tags",
    "additives_tags",
    "nova_group",
    "nutriscore_grade",
    "nutriscore_score",
    "nutriments",
    "countries_tags",
    "no_nutrition_data",
    "schema_version",
)


class OffSchemaError(RuntimeError):
    """Raised when the OFF Parquet export no longer has the columns we depend on."""


def _sql_quote(value: str) -> str:
    """Escape single quotes for embedding `value` inside a single-quoted SQL string literal."""
    return value.replace("'", "''")


def _check_schema(con: duckdb.DuckDBPyConnection, source_sql: str) -> None:
    """Verify the source Parquet still exposes every column we read.

    OFF regenerates this export nightly; if a future column rename/removal
    slips through it should surface as a clear error here rather than as
    silently NULL/missing data downstream.
    """
    # source_sql is a DuckDB read_parquet(...) call built from a config URL or a
    # local CLI-supplied path (see _sql_quote), not untrusted external input.
    described = con.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()  # noqa: S608
    available = {row[0] for row in described}
    missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in available]
    if missing:
        raise OffSchemaError(
            "Open Food Facts Parquet export is missing expected column(s): "
            f"{missing}. The upstream schema likely changed - update "
            "setup/open_food_facts/duckdb_creation.py's column mapping before retrying."
        )


def _create_macros(con: duckdb.DuckDBPyConnection) -> None:
    """Register small helper macros used by the extraction query below."""
    con.execute(
        """
        CREATE OR REPLACE MACRO lang_text(structs, lang_code) AS (
            list_filter(structs, x -> x.lang = lang_code)[1]['text']
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE MACRO nutrient_100g(nutriments, nutrient_name) AS (
            list_filter(nutriments, x -> x.name = nutrient_name)[1]['100g']::DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE MACRO nutrient_unit(nutriments, nutrient_name) AS (
            list_filter(nutriments, x -> x.name = nutrient_name)[1]['unit']
        )
        """
    )
    register_unit_ratio_macro(con)


# The `products` table schema, as (output column, source SQL expression) pairs.
# Single source of truth for both the build SELECT and the staleness check
# (`_expected_products_columns`), so adding a column here can't silently leave
# an old DB looking up to date. Nutrient columns are appended programmatically.
_BASE_SELECT: tuple[tuple[str, str], ...] = (
    ("code", "code"),
    ("product_name", "lang_text(product_name, 'en')"),
    ("product_name_pl", "lang_text(product_name, 'pl')"),
    ("ingredients_text", "COALESCE(lang_text(ingredients_text, 'en'), lang_text(ingredients_text, 'pl'))"),
    ("brands", "brands"),
    ("brands_tags", "brands_tags"),
    ("quantity", "quantity"),
    ("serving_size", "serving_size"),
    ("serving_quantity", "serving_quantity"),
    ("product_quantity", "product_quantity"),
    ("product_quantity_unit", "product_quantity_unit"),
    ("nutrition_data_per", "nutrition_data_per"),
    ("categories", "categories"),
    ("categories_tags", "categories_tags"),
    ("compared_to_category", "compared_to_category"),
    ("labels_tags", "labels_tags"),
    ("allergens_tags", "allergens_tags"),
    ("traces_tags", "traces_tags"),
    ("additives_tags", "additives_tags"),
    ("nova_group", "nova_group"),
    ("nutriscore_grade", "nutriscore_grade"),
    ("nutriscore_score", "nutriscore_score"),
)


def _expected_products_columns() -> set[str]:
    """The full set of columns a freshly built `products` table should have."""
    cols = {name for name, _ in _BASE_SELECT}
    for prefix in _NUTRIENT_COLUMNS.values():
        cols.add(stored_column(prefix))
    # Semantic-search vector, backfilled after the base table is materialized.
    cols.add("embedding")
    return cols


def _text(value: object) -> str | None:
    """Non-empty trimmed string, or None for null/blank values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _category_phrase(row: dict[str, Any]) -> str | None:
    """Most specific category label, humanized for prose (spaces, no lang prefix)."""
    compared = _text(row.get("compared_to_category"))
    if compared:
        return _TAG_PREFIX_RE.sub("", compared).replace("-", " ")
    categories = _text(row.get("categories"))
    if categories:
        # Hierarchy paths are comma-separated; the last segment is the leaf.
        return categories.split(",")[-1].strip() or None
    return None


def _indefinite_article(phrase: str) -> str:
    return "An" if phrase[:1].casefold() in "aeiou" else "A"


def _build_document(row: dict[str, Any]) -> str:
    """Per-product prose for the embedding model.

    Prose ("Diet Coke. A soda product by Coca-Cola.") encodes role relationships
    better than labelled lines. Only identity fields (name, category, brand) —
    not provenance, nutrients, ingredients, or labels. The trash filter guarantees
    a name or category, so the document is never empty.
    """
    name = _text(row.get("product_name"))
    name_pl = _text(row.get("product_name_pl"))
    if name_pl and name is not None and name_pl.casefold() == name.casefold():
        name_pl = None

    head = f"{name} ({name_pl})" if name and name_pl else name or name_pl
    category = _category_phrase(row)
    brands = _text(row.get("brands"))

    sentences: list[str] = []
    if head:
        sentences.append(f"{head}.")
    if category and brands:
        sentences.append(f"{_indefinite_article(category)} {category} product by {brands}.")
    elif category:
        sentences.append(f"{_indefinite_article(category)} {category} product.")
    elif brands:
        sentences.append(f"A product by {brands}.")
    if not sentences:
        raise ValueError("empty embedding document; trash filter should have kept a name or category")
    return " ".join(sentences)


def _embed_products(con: duckdb.DuckDBPyConnection, settings: SetupSettings) -> None:
    """Backfill the `embedding` column and build a VSS HNSW index over it."""
    embed_table(
        con,
        table="products",
        id_column="code",
        id_type="VARCHAR",
        columns=_DOCUMENT_COLUMNS,
        build_document=_build_document,
        dim=settings.off_embedding_dim,
        model_name=settings.off_embedding_model,
    )


def _select_columns() -> str:
    parts = [expr if expr == name else f"{expr} AS {name}" for name, expr in _BASE_SELECT]
    for off_key, prefix in _NUTRIENT_COLUMNS.items():
        amount = f"nutrient_100g(nutriments, '{off_key}')"
        unit = f"nutrient_unit(nutriments, '{off_key}')"
        parts.append(f"{scaled_amount_sql(amount, unit, prefix)} AS {stored_column(prefix)}")
    return ",\n           ".join(parts)


def _build_select_sql(source_sql: str) -> str:
    """Build the CREATE TABLE ... AS SELECT body.

    All interpolated fragments (`source_sql`, nutrient/macro keys) come from
    static config or the fixed `_BASE_SELECT`/`_NUTRIENT_COLUMNS`/`_REQUIRED_MACROS`
    mappings above, not from untrusted external input.

    The trash filter wraps the projection because `_trash_predicate()` is
    expressed over the extracted output columns (e.g. `product_name_pl`, the
    language-split text), which only exist once `_select_columns()` has run -
    the raw source still has them as per-language struct lists.
    """
    macro_filters = " AND\n      ".join(f"nutrient_100g(nutriments, '{key}') IS NOT NULL" for key in _REQUIRED_MACROS)
    return f"""
        SELECT * FROM (
            SELECT
               {_select_columns()}
            FROM {source_sql}
            WHERE list_contains(countries_tags, 'en:poland')
              AND (no_nutrition_data IS NULL OR no_nutrition_data = FALSE)
              AND code IS NOT NULL AND code != ''
              AND {macro_filters}
        )
        WHERE NOT {_trash_predicate()}
    """  # noqa: S608


def _column_comment_sql(table: str) -> list[str]:
    """`COMMENT ON COLUMN` statements documenting canonical units."""
    statements = [
        f"COMMENT ON COLUMN {table}.code IS 'Open Food Facts barcode (primary key).'",
        f"COMMENT ON COLUMN {table}.nutrition_data_per IS "
        "'OFF provenance flag: were nutrients reported per 100g directly, or derived from a serving size?'",
    ]
    for off_key, prefix in _NUTRIENT_COLUMNS.items():
        unit = TARGET_UNIT[prefix]
        col = stored_column(prefix)
        statements.append(f"COMMENT ON COLUMN {table}.{col} IS '{off_key} per 100g, in {unit} (canonical).'")
    return statements


def _stale_products_columns(target: Path) -> set[str] | None:
    """Expected `products` columns missing from the DB at `target`.

    Returns `None` when there is no usable DB to reuse (absent, empty, or
    unreadable) and an empty set when the existing DB already has every
    expected column. A non-empty set means the DB predates a schema change
    and should be rebuilt.
    """
    if not target.exists() or target.stat().st_size == 0:
        return None
    try:
        con = duckdb.connect(str(target), read_only=True)
    except duckdb.Error:
        return None
    try:
        described = con.execute("DESCRIBE products").fetchall()
    except duckdb.Error:
        return None
    finally:
        con.close()
    present = {row[0] for row in described}
    missing = _expected_products_columns() - present
    # Pre-canonical builds kept `<prefix>_unit` columns; their presence alone
    # must force a rebuild (expected - present is empty when only extras remain).
    stale_units = {c for c in present if c.endswith("_unit")}
    return missing | stale_units


def _load_vss(con: duckdb.DuckDBPyConnection) -> None:
    """Load the VSS extension so a table carrying a persisted HNSW index can be modified.

    DuckDB refuses to mutate `products` (e.g. the trash `DELETE`) unless the
    extension that provides its `HNSW` index type is loaded first.
    """
    con.execute("INSTALL vss")
    con.execute("LOAD vss")


def _create_fts_index(con: duckdb.DuckDBPyConnection) -> None:
    """(Re)build the BM25 full-text index over `_FTS_FIELDS`.

    `overwrite=1` drops and recreates only the `fts_main_products` schema, so
    this is safe to run in place on an existing DB without touching the
    `embedding` column or its HNSW index.
    """
    fields = ", ".join(f"'{f}'" for f in _FTS_FIELDS)
    con.execute(f"PRAGMA create_fts_index('products', 'code', {fields}, overwrite=1)")  # noqa: S608


def _existing_fts_fields(target: Path) -> set[str] | None:
    """The columns currently covered by the DB's FTS index, or `None` if absent.

    DuckDB records indexed fields in `fts_main_products.fields`. Returns `None`
    when there is no usable index to inspect (DB missing/unreadable, or no FTS
    index built yet) so the caller treats it the same as an out-of-date index.
    """
    if not target.exists() or target.stat().st_size == 0:
        return None
    try:
        con = duckdb.connect(str(target), read_only=True)
    except duckdb.Error:
        return None
    try:
        rows = con.execute("SELECT field FROM fts_main_products.fields").fetchall()
    except duckdb.Error:
        return None
    finally:
        con.close()
    return {row[0] for row in rows}


def _materialize_off_db(settings: SetupSettings, parquet_path: Path) -> Path:
    """Filter `parquet_path` into a fresh `products` DuckDB at `settings.off_db`.

    Writes to a sibling temp file and renames on success so an interrupted
    build never leaves a half-written DB in place. Assumes the Parquet is
    already available locally (see `resolve_source_parquet`).
    """
    target = settings.off_db
    source_sql = f"read_parquet('{_sql_quote(str(parquet_path))}')"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f".tmp-{os.getpid()}-{int(time.time())}{target.suffix}")

    try:
        con = duckdb.connect(str(tmp_path))
        try:
            _check_schema(con, source_sql)
            _create_macros(con)

            log.info("Filtering Open Food Facts products (source: %s)...", parquet_path)
            con.execute(f"CREATE TABLE products AS {_build_select_sql(source_sql)}")
            _clean_energy(con)
            row_count = con.execute("SELECT count(*) FROM products").fetchone()[0]  # type: ignore[index]

            for stmt in _column_comment_sql("products"):
                con.execute(stmt)

            try:
                _create_fts_index(con)
            except duckdb.Error as exc:
                log.warning("Could not build FTS index (falling back to ILIKE at query time): %s", exc)

            _embed_products(con, settings)

            con.execute(
                "CREATE TABLE off_meta "
                "(built_at TIMESTAMP, source_url VARCHAR, raw_parquet_path VARCHAR, row_count BIGINT, "
                "embedding_model VARCHAR)"
            )
            con.execute(
                "INSERT INTO off_meta VALUES (now(), ?, ?, ?, ?)",
                [settings.off_source_url, str(parquet_path), row_count, settings.off_embedding_model],
            )
        finally:
            con.close()
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    os.replace(tmp_path, target)
    log.info("Built OFF product DB at %s (%d Polish products with complete macros).", target, row_count)
    return target


def build_off_db(settings: SetupSettings) -> Path:
    """Ensure the local Polish OFF product DB exists and is current at `settings.off_db`.

    A DB predating a schema change (a column missing from `_BASE_SELECT`) is
    rebuilt from scratch: the OFF Parquet export is resolved to the local cache
    at `settings.off_raw_parquet` (downloaded once from `settings.off_source_url`
    when missing), then filtered to Polish products with a complete macro profile
    and materialized atomically. When the columns are already current the DB is
    instead upgraded in place, without the embedding pass: unidentifiable rows are
    deleted (see `_trash_predicate`), energy values are repaired/pruned (see
    `_clean_energy`), and the FTS index is rebuilt when rows went away or its
    field set is stale. So changing `_FTS_FIELDS`, the trash rules or the energy
    rules never forces a full re-embed; only a column change does.
    """
    target = settings.off_db
    missing = _stale_products_columns(target)
    if missing == set():
        fts_stale = _existing_fts_fields(target) != set(_FTS_FIELDS)
        con = duckdb.connect(str(target))
        try:
            _load_vss(con)
            deleted = con.execute(f"SELECT count(*) FROM products WHERE {_trash_predicate()}").fetchone()[0]  # type: ignore[index]  # noqa: S608
            if deleted:
                con.execute(f"DELETE FROM products WHERE {_trash_predicate()}")  # noqa: S608
                log.info("Removed %d unidentifiable OFF product(s) from %s.", deleted, target)
            deleted += _clean_energy(con)
            # Deleting rows shifts the BM25 corpus statistics, so the FTS index
            # is rebuilt whenever rows went away (as well as when its field set
            # is stale); surviving embeddings and the HNSW index stay valid.
            if deleted or fts_stale:
                _create_fts_index(con)
                if fts_stale:
                    log.info("Rebuilt stale FTS index at %s.", target)
            else:
                log.info("OFF product DB already present and current at %s, skipping build.", target)
        finally:
            con.close()
        return target
    if missing:
        log.info("OFF product DB at %s is missing expected column(s) %s; rebuilding.", target, sorted(missing))

    parquet_path = resolve_source_parquet(settings)
    return _materialize_off_db(settings, parquet_path)
