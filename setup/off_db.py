"""Build a local DuckDB database of Polish Open Food Facts products.

Source: the Open Food Facts Hugging Face Parquet export (already slimmed vs.
the raw MongoDB/JSONL dump). We filter down to products sold in Poland with a
complete macro profile, keep only the columns useful to the dietary advisor,
and materialize them into a single portable ``.duckdb`` file plus a full-text
index over product names/brands for lookup by name.

The full export is ~7.6 GB. We download it once to a local cache and then
filter from that file, rather than streaming it over ``httpfs`` on every run:
DuckDB's remote Parquet reads fan out into hundreds of ranged HTTP requests,
which Hugging Face rate-limits (HTTP 429). A single resumable download is both
friendlier to HF and far faster to re-run.

Nutrient amounts are stored exactly as reported by OFF (per 100g), alongside
their source unit, so unit canonicalization stays an explicit, later step
(the runtime `OffFoodDb` reader) rather than being baked into this build.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb

from dietary_advisor.config import Settings

log = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
_DOWNLOAD_LOG_EVERY_BYTES = 200 << 20  # log progress roughly every 200 MiB
_DOWNLOAD_TIMEOUT_S = 120.0
_DOWNLOAD_RETRIES = 5

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
    "quantity",
    "serving_size",
    "serving_quantity",
    "product_quantity",
    "product_quantity_unit",
    "nutrition_data_per",
    "categories_tags",
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


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GiB"


def _download_parquet(url: str, dest: Path) -> None:
    """Download `url` to `dest`, resuming a prior partial download when possible.

    Bytes accumulate in a sibling ``.part`` file (via HTTP Range requests) and
    are renamed onto `dest` only once complete, so an interrupted run never
    leaves a truncated file masquerading as the finished download. Transient
    failures (including HF's 429) are retried with exponential backoff.
    """
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to download from non-https URL: {url!r}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    last_exc: Exception | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "dietary-advisor-setup"})  # noqa: S310 (https-only)
        if resume_from:
            req.add_header("Range", f"bytes={resume_from}-")
        try:
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp:  # noqa: S310 (https-only)
                # A 200 (rather than 206) means the server ignored our Range
                # header - e.g. after a redirect dropped it - so start over.
                if resume_from and resp.status != 206:
                    resume_from = 0
                content_length = resp.headers.get("Content-Length")
                total = (int(content_length) + resume_from) if content_length else None
                downloaded = resume_from
                since_log = 0
                mode = "ab" if resume_from else "wb"
                if resume_from:
                    log.info("Resuming download at %s...", _human_bytes(resume_from))
                with part.open(mode) as fh:
                    while True:
                        chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        since_log += len(chunk)
                        if since_log >= _DOWNLOAD_LOG_EVERY_BYTES:
                            since_log = 0
                            if total:
                                log.info(
                                    "  downloaded %s / %s (%.0f%%)",
                                    _human_bytes(downloaded),
                                    _human_bytes(total),
                                    100 * downloaded / total,
                                )
                            else:
                                log.info("  downloaded %s", _human_bytes(downloaded))
            part.replace(dest)
            log.info("Download complete: %s (%s).", dest, _human_bytes(dest.stat().st_size))
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            wait = min(2**attempt, 30)
            log.warning("Download attempt %d/%d failed: %s. Retrying in %ds.", attempt, _DOWNLOAD_RETRIES, exc, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed to download OFF Parquet from {url} after {_DOWNLOAD_RETRIES} attempts: {last_exc}")


def _resolve_source_parquet(settings: Settings, raw_parquet: Path | None, redownload: bool) -> Path:
    """Return a local Parquet path to filter, downloading the OFF export to the cache if needed.

    An explicit `raw_parquet` is used as-is (must exist). Otherwise we use the
    cache at `settings.off_raw_parquet`, (re)downloading it from
    `settings.off_source_url` when missing or when `redownload` is set.
    """
    if raw_parquet is not None:
        if not raw_parquet.exists():
            raise FileNotFoundError(f"--raw-parquet file not found: {raw_parquet}")
        return raw_parquet

    cache = settings.off_raw_parquet
    if redownload and cache.exists():
        cache.unlink()
    if cache.exists() and cache.stat().st_size > 0:
        log.info("Using cached OFF Parquet at %s (%s).", cache, _human_bytes(cache.stat().st_size))
        return cache

    log.info("Downloading OFF Parquet (~7.6 GB) from %s to %s ...", settings.off_source_url, cache)
    _download_parquet(settings.off_source_url, cache)
    return cache


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
            "setup/off_db.py's column mapping before retrying."
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


def _nutrient_select_columns() -> str:
    parts = []
    for off_key, prefix in _NUTRIENT_COLUMNS.items():
        parts.append(f"nutrient_100g(nutriments, '{off_key}') AS {prefix}_100g")
        parts.append(f"nutrient_unit(nutriments, '{off_key}') AS {prefix}_unit")
    return ",\n       ".join(parts)


def _build_select_sql(source_sql: str) -> str:
    """Build the CREATE TABLE ... AS SELECT body.

    All interpolated fragments (`source_sql`, nutrient/macro keys) come from
    static config or the fixed `_NUTRIENT_COLUMNS`/`_REQUIRED_MACROS`
    mappings above, not from untrusted external input.
    """
    macro_filters = " AND\n      ".join(f"nutrient_100g(nutriments, '{key}') IS NOT NULL" for key in _REQUIRED_MACROS)
    return f"""
        SELECT
           code,
           lang_text(product_name, 'en') AS product_name,
           lang_text(product_name, 'pl') AS product_name_pl,
           lang_text(generic_name, 'en') AS generic_name,
           COALESCE(lang_text(ingredients_text, 'en'), lang_text(ingredients_text, 'pl')) AS ingredients_text,
           brands,
           quantity,
           serving_size,
           serving_quantity,
           product_quantity,
           product_quantity_unit,
           nutrition_data_per,
           categories_tags,
           labels_tags,
           allergens_tags,
           traces_tags,
           additives_tags,
           nova_group,
           nutriscore_grade,
           nutriscore_score,
           {_nutrient_select_columns()}
        FROM {source_sql}
        WHERE list_contains(countries_tags, 'en:poland')
          AND (no_nutrition_data IS NULL OR no_nutrition_data = FALSE)
          AND code IS NOT NULL AND code != ''
          AND {macro_filters}
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


def build_off_db(
    settings: Settings,
    *,
    force: bool = False,
    raw_parquet: Path | None = None,
    redownload: bool = False,
) -> Path:
    """Ensure the local Polish OFF product DB exists at `settings.off_db`.

    If it already exists (and `force` is False), this is a no-op and the
    existing path is returned unchanged. Otherwise the OFF Parquet export is
    resolved to a local file - the cache at `settings.off_raw_parquet`
    (downloaded once from `settings.off_source_url` when missing), or an
    explicit `raw_parquet` - then filtered to Polish products with a complete
    macro profile and materialized into a fresh `.duckdb` file, written
    atomically.
    """
    target = settings.off_db
    if target.exists() and target.stat().st_size > 0 and not force:
        log.info("OFF product DB already present at %s, skipping build (use --force to rebuild).", target)
        return target

    parquet_path = _resolve_source_parquet(settings, raw_parquet, redownload)
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
                con.execute(
                    "PRAGMA create_fts_index('products', 'code', 'product_name', 'product_name_pl', 'brands', "
                    "overwrite=1)"
                )
            except duckdb.Error as exc:
                log.warning("Could not build FTS index (falling back to ILIKE at query time): %s", exc)

            con.execute(
                "CREATE TABLE off_meta "
                "(built_at TIMESTAMP, source_url VARCHAR, raw_parquet_path VARCHAR, row_count BIGINT)"
            )
            con.execute(
                "INSERT INTO off_meta VALUES (now(), ?, ?, ?)",
                [settings.off_source_url, str(parquet_path), row_count],
            )
        finally:
            con.close()
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    os.replace(tmp_path, target)
    log.info("Built OFF product DB at %s (%d Polish products with complete macros).", target, row_count)
    return target
