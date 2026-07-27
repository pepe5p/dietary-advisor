"""Read models for the two underlying food databases.

`OFFItem`/`USDAItem` mirror the raw DuckDB row shape 1:1 (minus the embedding
vector). Unlike the domain `dietary_advisor.totaller.nutrition.FoodItem`, they
are read-only, never emitted by the LLM, and deliberately do not share a base
class - each DB has different provenance/context fields.
`dietary_advisor.planning.hydration.to_food_item` is the only place either
becomes a `FoodItem`; LLM-facing search hits are built by
`OFFHit.create_from_off_item` / `USDAHit.create_from_usda_item`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# A raw DuckDB row keyed by column name. The column set varies per query (and is
# open-ended under `SELECT *`), so it is deliberately not a fixed model.
type Row = dict[str, Any]


class OFFItem(BaseModel):
    """A product row as stored in the Open Food Facts DuckDB.

    `code` is the raw barcode; runtime codes are `off:<barcode>` (see
    `off_food_db.to_code`) so they never collide with USDA FDC ids.
    """

    model_config = ConfigDict(extra="ignore")

    code: str = Field(description="Open Food Facts barcode.")
    product_name: str | None = None
    product_name_pl: str | None = None
    ingredients_text: str | None = None
    brands: str | None = None
    brands_tags: list[str] = Field(default_factory=list)
    quantity: str | None = None
    serving_size: str | None = None
    serving_quantity: float | None = None
    product_quantity: float | None = None
    product_quantity_unit: str | None = None
    nutrition_data_per: str | None = None
    categories: str | None = None
    categories_tags: list[str] = Field(default_factory=list)
    compared_to_category: str | None = None
    labels_tags: list[str] = Field(default_factory=list)
    allergens_tags: list[str] = Field(default_factory=list)
    traces_tags: list[str] = Field(default_factory=list)
    additives_tags: list[str] = Field(default_factory=list)

    @field_validator(
        "brands_tags",
        "categories_tags",
        "labels_tags",
        "allergens_tags",
        "traces_tags",
        "additives_tags",
        mode="before",
    )
    @classmethod
    def _coerce_null_tag_lists(cls, value: object) -> object:
        # DuckDB returns SQL NULL for empty list columns; treat as [].
        return [] if value is None else value

    nova_group: float | None = None
    nutriscore_grade: str | None = None
    nutriscore_score: float | None = None

    energy_kcal_in_100g: float | None = None
    energy_kj_in_100g: float | None = None
    proteins_g_in_100g: float | None = None
    carbohydrates_g_in_100g: float | None = None
    sugars_g_in_100g: float | None = None
    fat_g_in_100g: float | None = None
    saturated_fat_g_in_100g: float | None = None
    fiber_g_in_100g: float | None = None
    salt_g_in_100g: float | None = None
    sodium_mg_in_100g: float | None = None
    potassium_mg_in_100g: float | None = None
    calcium_mg_in_100g: float | None = None
    iron_mg_in_100g: float | None = None
    vitamin_c_mg_in_100g: float | None = None
    vitamin_d_ug_in_100g: float | None = None
    cholesterol_mg_in_100g: float | None = None


class USDAItem(BaseModel):
    """A food row as stored in the USDA FoodData Central DuckDB.

    `fdc_id` is the raw primary key; runtime codes are `usda:<fdc_id>` (see
    `usda_food_db.to_code`) so they never collide with OFF barcodes.
    """

    model_config = ConfigDict(extra="ignore")

    fdc_id: int
    description: str
    category: str | None = None

    energy_kcal_in_100g: float | None = None
    proteins_g_in_100g: float | None = None
    carbohydrates_g_in_100g: float | None = None
    fat_g_in_100g: float | None = None
    saturated_fat_g_in_100g: float | None = None
    fiber_g_in_100g: float | None = None
    sugars_g_in_100g: float | None = None
    sodium_mg_in_100g: float | None = None
    potassium_mg_in_100g: float | None = None
    calcium_mg_in_100g: float | None = None
    iron_mg_in_100g: float | None = None
    vitamin_c_mg_in_100g: float | None = None
    vitamin_d_ug_in_100g: float | None = None
    cholesterol_mg_in_100g: float | None = None
