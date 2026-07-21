"""One-off migration: canonicalize nutrient units and rename columns (no re-embed).

For each nutrient prefix:
1. If a legacy `<prefix>_unit` column exists, rescale `<prefix>_100g` into
   `TARGET_UNIT[prefix]` and drop the unit column.
2. Rename `<prefix>_100g` to `stored_column(prefix)` (e.g. `sodium_mg_in_100g`).

The `embedding` column is untouched; the HNSW index is dropped and rebuilt from
the existing vectors (no re-embed). FTS indexes are unaffected.

Idempotent: unit scaling runs only while `_unit` columns exist; renames skip
when the target column name is already present.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import duckdb

from dietary_advisor.config import get_settings
from setup.embedding import _build_hnsw
from setup.open_food_facts.duckdb_creation import _NUTRIENT_COLUMNS
from setup.units import register_unit_ratio_macro, stored_column, TARGET_UNIT
from setup.usda.duckdb_creation import _NUTRIENT_NUMBERS

log = logging.getLogger(__name__)


def _backup(path: Path) -> None:
    dest = path.with_suffix(path.suffix + ".bak")
    if dest.exists():
        log.info("Backup already exists at %s", dest)
        return
    shutil.copy2(path, dest)
    log.info("Backed up %s -> %s", path, dest)


def _migrate(path: Path, table: str, prefixes: list[str]) -> None:
    if not path.exists():
        log.warning("Skipping %s: not found", path)
        return
    _backup(path)
    con = duckdb.connect(str(path))
    try:
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        con.execute("SET hnsw_enable_experimental_persistence = true")
        register_unit_ratio_macro(con)

        present = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}  # noqa: S608
        unit_pending = [p for p in prefixes if f"{p}_unit" in present]
        rename_pending = [p for p in prefixes if f"{p}_100g" in present and stored_column(p) not in present]
        if not unit_pending and not rename_pending:
            log.info("%s: already migrated, nothing to do", path)
            return

        # DuckDB refuses to drop/rename columns before an indexed trailing column.
        con.execute(f"DROP INDEX IF EXISTS {table}_embedding_hnsw")  # noqa: S608

        for prefix in unit_pending:
            legacy_col = f"{prefix}_100g"
            unit_col = f"{prefix}_unit"
            target = TARGET_UNIT[prefix]
            con.execute(
                f"UPDATE {table} SET {legacy_col} = {legacy_col} * unit_ratio({unit_col}, '{target}')"  # noqa: S608
            )
            con.execute(f"ALTER TABLE {table} DROP COLUMN {unit_col}")  # noqa: S608
            log.info("%s.%s -> %s (dropped %s)", table, prefix, target, unit_col)

        for prefix in rename_pending:
            legacy_col = f"{prefix}_100g"
            new_col = stored_column(prefix)
            con.execute(f"ALTER TABLE {table} RENAME COLUMN {legacy_col} TO {new_col}")  # noqa: S608
            log.info("%s: renamed %s -> %s", table, legacy_col, new_col)

        _build_hnsw(con, table)
        log.info(
            "%s: unit-migrated %d, renamed %d, HNSW rebuilt",
            path,
            len(unit_pending),
            len(rename_pending),
        )
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    _migrate(settings.off_db, "products", list(_NUTRIENT_COLUMNS.values()))
    _migrate(settings.usda_db, "foods", list(_NUTRIENT_NUMBERS))


if __name__ == "__main__":
    main()
