"""Setup-time document construction for product embeddings."""

from __future__ import annotations

import duckdb

from setup.open_food_facts.duckdb_creation import _document_sql, _expected_products_columns, _trash_predicate


def _document_for(**columns: object) -> str:
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE products ("
            "product_name VARCHAR, product_name_pl VARCHAR, generic_name VARCHAR, brands VARCHAR, "
            "categories VARCHAR, compared_to_category VARCHAR, labels_tags VARCHAR[], ingredients_text VARCHAR)"
        )
        keys = list(columns)
        placeholders = ", ".join("?" for _ in keys)
        insert = f"INSERT INTO products ({', '.join(keys)}) VALUES ({placeholders})"  # noqa: S608 (test-only; hardcoded kwargs)
        con.execute(insert, list(columns.values()))
        return con.execute(f"SELECT {_document_sql()} FROM products").fetchone()[0]  # type: ignore[index]  # noqa: S608
    finally:
        con.close()


def test_document_labels_every_valuable_column_and_strips_tag_prefixes() -> None:
    doc = _document_for(
        product_name="Mozzarella",
        product_name_pl="Mozzarella light",
        generic_name="soft cheese",
        brands="Galbani",
        categories="Cheeses",
        compared_to_category="en:mozzarella",
        labels_tags=["en:vegetarian", "en:organic"],
        ingredients_text="milk, salt",
    )
    assert doc == (
        "Name: Mozzarella\n"
        "Nazwa: Mozzarella light\n"
        "Description: soft cheese\n"
        "Brand: Galbani\n"
        "Categories: Cheeses\n"
        "Category: mozzarella\n"
        "Labels: vegetarian, organic\n"
        "Ingredients: milk, salt"
    )


def test_document_skips_missing_parts() -> None:
    doc = _document_for(product_name_pl="Jogurt naturalny", labels_tags=[])
    assert doc == "Nazwa: Jogurt naturalny"


def test_document_truncates_long_ingredients() -> None:
    doc = _document_for(product_name="X", ingredients_text="a" * 1000)
    ingredients_line = doc.splitlines()[-1]
    assert ingredients_line.startswith("Ingredients: ")
    assert len(ingredients_line) == len("Ingredients: ") + 600


def test_expected_columns_include_embedding() -> None:
    assert "embedding" in _expected_products_columns()


def _codes_surviving_trash_filter(rows: list[dict[str, object]]) -> set[str]:
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE products ("
            "code VARCHAR, product_name VARCHAR, product_name_pl VARCHAR, generic_name VARCHAR, "
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
            {"code": "name-only", "product_name": "Duck fillet"},
            {"code": "pl-name-only", "product_name_pl": "Filet z kaczki"},
            {"code": "generic-only", "generic_name": "soft cheese"},
            {"code": "category-only", "categories": "Ducks"},
            {"code": "category-tag-only", "categories_tags": ["en:ducks"]},
            {"code": "compared-only", "compared_to_category": "en:duck-fillets"},
            {"code": "ingredients-only", "ingredients_text": "milk, salt"},
        ]
    )
    assert survivors == {
        "name-only",
        "pl-name-only",
        "generic-only",
        "category-only",
        "category-tag-only",
        "compared-only",
        "ingredients-only",
    }
