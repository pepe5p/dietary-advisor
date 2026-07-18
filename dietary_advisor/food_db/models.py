"""Read models for the two underlying food databases.

`OFFItem`/`USDAItem` mirror the raw shape of each DuckDB source; unlike the
domain `dietary_advisor.schemas.nutrition.FoodItem`, they are read-only,
never emitted by the LLM, and deliberately do not share a base class - each
DB has different provenance/context fields, and forcing a common shape would
either lose information or fabricate fields the source doesn't have.
`dietary_advisor.hydration.to_food_item` is the only place either becomes a
`FoodItem`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.schemas.nutrition import NutrientAmountMap


class OFFItem(BaseModel):
    """A product as read from the Open Food Facts DB."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(description="Open Food Facts barcode.")
    name: str
    description: str | None = None
    nutrients_per_100g: NutrientAmountMap = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    brands: str | None = None
    brands_tags: list[str] = Field(default_factory=list)
    categories: str | None = None
    categories_tags: list[str] = Field(default_factory=list)
    compared_to_category: str | None = None
    ingredients_text: str | None = None


class USDAItem(BaseModel):
    """A food as read from the USDA FoodData Central DB.

    The complement to `OFFItem`: generic/whole foods (Foundation Foods and SR
    Legacy) that OFF's branded catalogue largely lacks. `code` is the FDC id
    prefixed with `usda:` (see `usda_food_db.to_code`) so it can never
    collide with an OFF barcode.
    """

    model_config = ConfigDict(extra="ignore")

    code: str
    name: str
    description: str | None = None
    nutrients_per_100g: NutrientAmountMap = Field(default_factory=dict)
    category: str | None = None
    scientific_name: str | None = None
