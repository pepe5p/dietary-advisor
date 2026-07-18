"""Build a local DuckDB database of Polish Open Food Facts products.

Source: the Open Food Facts Hugging Face Parquet export (already slimmed vs.
the raw MongoDB/JSONL dump), fetched to a local cache by
``setup.off_downloading``. We filter down to products sold in Poland with a
complete macro profile, drop unidentifiable rows (see ``_trash_predicate``), keep
only the columns useful to the dietary advisor, and materialize them into a
single portable ``.duckdb`` file plus a full-text index over product
names/brands/categories (see ``_FTS_FIELDS``) for lookup by name and
descriptive text.

Nutrient amounts are stored exactly as reported by OFF (per 100g), alongside
their source unit, so unit canonicalization stays an explicit, later step
(the runtime `OffFoodDb` reader) rather than being baked into this build.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import duckdb

from dietary_advisor.config import Settings
from dietary_advisor.food_db.embeddings import load_embedder
from setup.off_downloading import resolve_source_parquet

log = logging.getLogger(__name__)

# Ingredients lists can be very long; cap them so they don't crowd out the
# name/category signal within the embedding model's ~512-token window.
_INGREDIENTS_MAX_CHARS = 600

# E5 document-side prefix. The query-side counterpart lives in
# dietary_advisor.food_db.embeddings; both share load_embedder so the two
# vector sets are comparable.
_PASSAGE_PREFIX = "passage: "

# Columns fed to the BM25 full-text index. Beyond names/brands this includes
# `categories`/`generic_name`/`ingredients_text` so lexical search can hit the
# English descriptive text of a product whose name is only in another language
# (e.g. the Polish "Filet z kaczki" is reachable via its "duck" category).
# Single source of truth: the build creates the index from this, and
# `build_off_db` compares it against an existing DB's indexed fields to decide
# whether an in-place FTS-only rebuild is needed. `categories_tags` is left out
# deliberately - it's a VARCHAR[] the FTS indexer rejects, and its content
# duplicates `categories`.
_FTS_FIELDS: tuple[str, ...] = (
    "product_name",
    "product_name_pl",
    "brands",
    "generic_name",
    "categories",
)

# Columns that identify *what a food is* - a product name, a generic
# description, a category, or an ingredient list. A row with all of these empty
# cannot be identified or matched by search even if it still carries a brand or
# label/allergen tags (e.g. barcode 5900766003084: brand "Polskie młyny" but no
# name, category or ingredients - useless to the agent), so setup drops it.
# Brand and label/allergen/trace tags are deliberately excluded: they never say
# what the product actually is. The two groups differ only in the emptiness
# test (blank string vs. zero-length list).
_IDENTIFYING_TEXT_COLUMNS: tuple[str, ...] = (
    "product_name",
    "product_name_pl",
    "generic_name",
    "categories",
    "compared_to_category",
    "ingredients_text",
)
_IDENTIFYING_LIST_COLUMNS: tuple[str, ...] = ("categories_tags",)


def _trash_predicate() -> str:
    """SQL boolean, true for rows with no identifying content (see `_IDENTIFYING_*_COLUMNS`).

    Shared by the fresh-build filter and the in-place cleanup so both agree on
    exactly which rows are worthless.
    """
    text = [f"NULLIF(trim({c}), '') IS NULL" for c in _IDENTIFYING_TEXT_COLUMNS]
    lists = [f"COALESCE(len({c}), 0) = 0" for c in _IDENTIFYING_LIST_COLUMNS]
    return "(" + " AND ".join(text + lists) + ")"


def _embed_documents(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """Embed product documents (build-time only) with the E5 `passage:` prefix."""
    prefixed = [_PASSAGE_PREFIX + t for t in texts]
    return [vec.tolist() for vec in load_embedder().embed(prefixed, batch_size=batch_size)]


# OFF nutrient key (as found in the `nutriments` struct list) -> our column
# prefix. Values are stored per-100g, in whatever unit OFF reports (captured
# alongside in a sibling `<prefix>_unit` column).
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
    "generic_name",
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
            "setup/duckdb_creation.py's column mapping before retrying."
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


# The `products` table schema, as (output column, source SQL expression) pairs.
# Single source of truth for both the build SELECT and the staleness check
# (`_expected_products_columns`), so adding a column here can't silently leave
# an old DB looking up to date. Nutrient columns are appended programmatically.
_BASE_SELECT: tuple[tuple[str, str], ...] = (
    ("code", "code"),
    ("product_name", "lang_text(product_name, 'en')"),
    ("product_name_pl", "lang_text(product_name, 'pl')"),
    ("generic_name", "lang_text(generic_name, 'en')"),
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
        cols.add(f"{prefix}_100g")
        cols.add(f"{prefix}_unit")
    # Semantic-search vector, backfilled after the base table is materialized.
    cols.add("embedding")
    return cols


def _document_sql() -> str:
    """SQL expression producing the per-product "what this product is" document.

    Each valuable column becomes a labelled line; NULL/empty parts are dropped
    by `concat_ws`. This single text is what gets embedded, so it must line up
    with the columns a user would describe a product by (name, brand, category,
    ingredients) rather than provenance/nutrient bookkeeping.
    """
    tags = "list_transform(labels_tags, x -> regexp_replace(x, '^[a-z]{2,3}:', ''))"
    compared = "regexp_replace(compared_to_category, '^[a-z]{2,3}:', '')"
    parts = (
        "CASE WHEN product_name IS NOT NULL THEN 'Name: ' || product_name END",
        "CASE WHEN product_name_pl IS NOT NULL THEN 'Nazwa: ' || product_name_pl END",
        "CASE WHEN generic_name IS NOT NULL THEN 'Description: ' || generic_name END",
        "CASE WHEN brands IS NOT NULL THEN 'Brand: ' || brands END",
        "CASE WHEN categories IS NOT NULL THEN 'Categories: ' || categories END",
        f"CASE WHEN compared_to_category IS NOT NULL THEN 'Category: ' || {compared} END",
        f"CASE WHEN len(labels_tags) > 0 THEN 'Labels: ' || array_to_string({tags}, ', ') END",
        "CASE WHEN ingredients_text IS NOT NULL "
        f"THEN 'Ingredients: ' || left(ingredients_text, {_INGREDIENTS_MAX_CHARS}) END",
    )
    return "concat_ws(chr(10), " + ", ".join(parts) + ")"


def _embed_products(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Backfill the `embedding` column and build a VSS HNSW index over it.

    Runs one embedding pass over every product document; the HNSW build is
    best-effort (the runtime falls back to brute-force cosine, cheap at this
    row count) so a missing/failed VSS extension never blocks the build.
    """
    dim = settings.off_embedding_dim
    con.execute(f"ALTER TABLE products ADD COLUMN embedding FLOAT[{dim}]")

    rows = con.execute(f"SELECT code, {_document_sql()} AS document FROM products").fetchall()  # noqa: S608
    if not rows:
        return
    codes = [r[0] for r in rows]
    documents = [r[1] or "" for r in rows]

    log.info("Embedding %d product documents with %s...", len(documents), settings.off_embedding_model)
    vectors = _embed_documents(documents)

    con.execute(f"CREATE TEMP TABLE _embeddings (code VARCHAR, embedding FLOAT[{dim}])")
    con.executemany("INSERT INTO _embeddings VALUES (?, ?)", list(zip(codes, vectors, strict=True)))
    con.execute(
        "UPDATE products SET embedding = _embeddings.embedding FROM _embeddings WHERE products.code = _embeddings.code"
    )
    con.execute("DROP TABLE _embeddings")

    try:
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        # Required to create an HNSW index in a disk-backed DB. Safe here: the
        # artifact is written atomically and only ever opened read-only after.
        con.execute("SET hnsw_enable_experimental_persistence = true")
        con.execute("CREATE INDEX products_embedding_hnsw ON products USING HNSW (embedding) WITH (metric = 'cosine')")
    except duckdb.Error as exc:
        log.warning("Could not build VSS/HNSW index (falling back to brute-force cosine at query time): %s", exc)


def _select_columns() -> str:
    parts = [expr if expr == name else f"{expr} AS {name}" for name, expr in _BASE_SELECT]
    for off_key, prefix in _NUTRIENT_COLUMNS.items():
        parts.append(f"nutrient_100g(nutriments, '{off_key}') AS {prefix}_100g")
        parts.append(f"nutrient_unit(nutriments, '{off_key}') AS {prefix}_unit")
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
    """`COMMENT ON COLUMN` statements documenting native OFF units, for anyone browsing the DB directly."""
    statements = [
        f"COMMENT ON COLUMN {table}.code IS 'Open Food Facts barcode (primary key).'",
        f"COMMENT ON COLUMN {table}.nutrition_data_per IS "
        "'OFF provenance flag: were nutrients reported per 100g directly, or derived from a serving size?'",
    ]
    for off_key, prefix in _NUTRIENT_COLUMNS.items():
        statements.append(
            f"COMMENT ON COLUMN {table}.{prefix}_100g IS "
            f"'{off_key} per 100g, in the unit reported by OFF (see {prefix}_unit; not yet canonicalized).'"
        )
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
    return _expected_products_columns() - present


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


def _materialize_off_db(settings: Settings, parquet_path: Path) -> Path:
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


def build_off_db(settings: Settings) -> Path:
    """Ensure the local Polish OFF product DB exists and is current at `settings.off_db`.

    A DB predating a schema change (a column missing from `_BASE_SELECT`) is
    rebuilt from scratch: the OFF Parquet export is resolved to the local cache
    at `settings.off_raw_parquet` (downloaded once from `settings.off_source_url`
    when missing), then filtered to Polish products with a complete macro profile
    and materialized atomically. When the columns are already current the DB is
    instead upgraded in place, without the embedding pass: unidentifiable rows are
    deleted (see `_trash_predicate`) and the FTS index is rebuilt when rows went
    away or its field set is stale. So changing `_FTS_FIELDS` or the trash rules
    never forces a full re-embed; only a column change does.
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
