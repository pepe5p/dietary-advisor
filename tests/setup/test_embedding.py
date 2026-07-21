"""Streaming embed_table over an in-memory DuckDB with a fake embedder."""

from __future__ import annotations

from typing import Any

import duckdb
import pytest

import setup.embedding as emb


def test_embed_table_streams_and_populates_every_row(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list[str], batch_size: int = 256) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    monkeypatch.setattr(emb, "embed_documents", fake_embed)
    monkeypatch.setattr(emb, "_ROW_BATCH", 2)
    monkeypatch.setattr(emb, "_build_hnsw", lambda con, table: None)

    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE products (code VARCHAR, product_name VARCHAR)")
        con.executemany(
            "INSERT INTO products VALUES (?, ?)",
            [("1", "Apple"), ("2", "Banana"), ("3", "Cherry")],
        )

        def build_document(row: dict[str, Any]) -> str:
            return f"Name: {row['product_name']}"

        emb.embed_table(
            con,
            table="products",
            id_column="code",
            id_type="VARCHAR",
            columns=("product_name",),
            build_document=build_document,
            dim=3,
            model_name="fake-model",
        )

        rows = con.execute("SELECT code, product_name, embedding FROM products ORDER BY code").fetchall()
        assert len(rows) == 3
        for code, name, vector in rows:
            assert vector is not None
            assert len(vector) == 3
            # "Name: {name}" length, as returned by the fake embedder.
            assert vector[0] == float(len(f"Name: {name}"))
            assert code in {"1", "2", "3"}
    finally:
        con.close()
