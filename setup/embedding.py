"""Build-time document embedding shared by the OFF and USDA DuckDB builds.

The runtime query embedder lives in `dietary_advisor.food_db.embeddings`; both
sides share `load_embedder` so their vectors land in the same space.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import duckdb

from dietary_advisor.food_db.embeddings import load_embedder

log = logging.getLogger(__name__)

# Rows streamed from DuckDB per embed batch. Keeps peak memory bounded while
# still feeding the embedder large enough chunks to amortize ONNX overhead.
_ROW_BATCH = 512


def embed_documents(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """Embed product documents (build-time only)."""
    return [vec.tolist() for vec in load_embedder().embed(texts, batch_size=batch_size)]


def _build_hnsw(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Best-effort VSS HNSW index over `table.embedding`.

    The runtime falls back to brute-force cosine (cheap at current row counts),
    so a missing/failed VSS extension never blocks the build.
    """
    try:
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        # Required to create an HNSW index in a disk-backed DB. Safe here: the
        # artifact is written atomically and only ever opened read-only after.
        con.execute("SET hnsw_enable_experimental_persistence = true")
        con.execute(
            f"CREATE INDEX {table}_embedding_hnsw ON {table} "  # noqa: S608
            f"USING HNSW (embedding) WITH (metric = 'cosine')"
        )
    except duckdb.Error as exc:
        log.warning(
            "Could not build VSS/HNSW index on %s (falling back to brute-force cosine at query time): %s",
            table,
            exc,
        )


def embed_table(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    id_column: str,
    id_type: str,
    columns: Sequence[str],
    build_document: Callable[[dict[str, Any]], str],
    dim: int,
    model_name: str,
) -> None:
    """Backfill `table.embedding` by streaming rows, embedding in batches, then building HNSW.

    A separate cursor streams the read so the main connection can write the
    temp embedding table without materializing the whole corpus in memory.
    `table`/`id_column`/`id_type`/`columns` come from static build config, not
    untrusted input.
    """
    con.execute(f"ALTER TABLE {table} ADD COLUMN embedding FLOAT[{dim}]")  # noqa: S608
    total = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # type: ignore[index]  # noqa: S608
    if not total:
        return

    log.info("Starting embedding of %d %s documents with %s", total, table, model_name)
    con.execute(f"CREATE TEMP TABLE _embeddings ({id_column} {id_type}, embedding FLOAT[{dim}])")  # noqa: S608
    reader = con.cursor()
    col_list = ", ".join(columns)
    reader.execute(f"SELECT {id_column}, {col_list} FROM {table}")  # noqa: S608
    keys = (id_column, *columns)
    done = 0
    reported_pct = 0
    while batch := reader.fetchmany(_ROW_BATCH):
        rows = [dict(zip(keys, r, strict=True)) for r in batch]
        vectors = embed_documents([build_document(r) for r in rows])
        con.executemany(
            "INSERT INTO _embeddings VALUES (?, ?)",
            [(r[id_column], v) for r, v in zip(rows, vectors, strict=True)],
        )
        done += len(rows)
        pct = done * 100 // total
        while reported_pct + 10 <= pct:
            reported_pct += 10
            log.info("Embedding %s: %d%% (%d/%d)", table, reported_pct, done, total)
    if reported_pct < 100:
        log.info("Embedding %s: 100%% (%d/%d)", table, done, total)
    con.execute(
        f"UPDATE {table} SET embedding = _embeddings.embedding "  # noqa: S608
        f"FROM _embeddings WHERE {table}.{id_column} = _embeddings.{id_column}"
    )
    con.execute("DROP TABLE _embeddings")
    _build_hnsw(con, table)
