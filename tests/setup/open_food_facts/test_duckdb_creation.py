"""Setup-time document construction for product embeddings."""

from __future__ import annotations

import duckdb
import pytest

from setup.open_food_facts.duckdb_creation import _build_document, _expected_products_columns, _trash_predicate


def test_document_prose_with_name_category_and_brand() -> None:
    doc = _build_document(
        {
            "product_name": "Mozzarella",
            "product_name_pl": "Mozzarella light",
            "brands": "Galbani",
            "categories": "Cheeses",
            "compared_to_category": "en:mozzarella",
        }
    )
    assert doc == "Mozzarella (Mozzarella light). A mozzarella product by Galbani."


def test_document_skips_missing_and_blank_parts() -> None:
    doc = _build_document({"product_name_pl": "Jogurt naturalny", "brands": "   "})
    assert doc == "Jogurt naturalny."


def test_document_dedupes_identical_polish_name() -> None:
    doc = _build_document({"product_name": "Pipe rigate", "product_name_pl": "Pipe Rigate"})
    assert doc == "Pipe rigate."


def test_document_falls_back_to_categories_leaf() -> None:
    doc = _build_document(
        {
            "product_name": "Breadsticks",
            "categories": "Snacks, Salty snacks, Breads, Breadsticks",
        }
    )
    assert doc == "Breadsticks. A Breadsticks product."


def test_document_brand_only_clause() -> None:
    doc = _build_document({"product_name": "Diet Coke", "brands": "Coca-Cola"})
    assert doc == "Diet Coke. A product by Coca-Cola."


def test_document_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty embedding document"):
        _build_document({"brands": "   "})


def test_expected_columns_include_embedding_without_unit_columns() -> None:
    cols = _expected_products_columns()
    assert "embedding" in cols
    assert "generic_name" not in cols
    assert "sodium_unit" not in cols
    assert "vitamin_d_unit" not in cols
    assert "sodium_mg_in_100g" in cols
    assert "vitamin_d_ug_in_100g" in cols


def _codes_surviving_trash_filter(rows: list[dict[str, object]]) -> set[str]:
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE products ("
            "code VARCHAR, product_name VARCHAR, product_name_pl VARCHAR, "
            "brands VARCHAR, categories VARCHAR, compared_to_category VARCHAR, ingredients_text VARCHAR, "
            "brands_tags VARCHAR[], categories_tags VARCHAR[], labels_tags VARCHAR[], "
            "allergens_tags VARCHAR[], traces_tags VARCHAR[])"
        )
        for row in rows:
            keys = list(row)
            placeholders = ", ".join("?" for _ in keys)
            insert = f"INSERT INTO products ({', '.join(keys)}) VALUES ({placeholders})"  # noqa: S608 (test-only)
            con.execute(insert, list(row.values()))
        con.execute(f"DELETE FROM products WHERE {_trash_predicate()}")  # noqa: S608 (predicate is static)
        return {r[0] for r in con.execute("SELECT code FROM products").fetchall()}
    finally:
        con.close()


def test_trash_filter_keeps_only_identifiable_rows() -> None:
    survivors = _codes_surviving_trash_filter(
        [
            {"code": "5901597860723"},  # barcode + nutrients only: dropped
            {"code": "brand-only", "brands": "Polskie młyny", "brands_tags": ["xx:polskie-mlyny"]},  # dropped
            {"code": "label-only", "labels_tags": ["en:organic"]},  # not identifying: dropped
            {"code": "allergen-only", "allergens_tags": ["en:milk"]},  # not identifying: dropped
            {"code": "blank-name", "product_name": "   "},  # whitespace counts as empty: dropped
            {"code": "ingredients-only", "ingredients_text": "milk, salt"},  # not in document: dropped
            {"code": "category-tag-only", "categories_tags": ["en:ducks"]},  # tags alone: dropped
            {"code": "name-only", "product_name": "Duck fillet"},
            {"code": "pl-name-only", "product_name_pl": "Filet z kaczki"},
            {"code": "category-only", "categories": "Ducks"},
            {"code": "compared-only", "compared_to_category": "en:duck-fillets"},
        ]
    )
    assert survivors == {
        "name-only",
        "pl-name-only",
        "category-only",
        "compared-only",
    }
