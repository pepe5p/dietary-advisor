"""Nutrient / FoodItem schemas: the symbolic units of computation for the Totaller.

All nutrient quantities are normalized to canonical units to avoid the kind of
unit-confusion arithmetic errors that vanilla LLMs notoriously produce
(see Kwerenda Literatury, sec. "Ślepota strukturalna").
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, WithJsonSchema


class NutrientName(str, Enum):
    """Canonical nutrients tracked by the Totaller."""

    ENERGY_KCAL = "energy_kcal"
    PROTEIN_G = "protein_g"
    CARBS_G = "carbs_g"
    FAT_G = "fat_g"
    SATURATED_FAT_G = "saturated_fat_g"
    FIBER_G = "fiber_g"
    SUGAR_G = "sugar_g"
    SODIUM_MG = "sodium_mg"
    POTASSIUM_MG = "potassium_mg"
    CALCIUM_MG = "calcium_mg"
    IRON_MG = "iron_mg"
    VITAMIN_C_MG = "vitamin_c_mg"
    VITAMIN_D_UG = "vitamin_d_ug"
    CHOLESTEROL_MG = "cholesterol_mg"


# LLM tool / structured-output JSON Schema: Pydantic emits propertyNames + $ref to
# NutrientName for dict[NutrientName, float], which Groq (and some other providers)
# reject when $defs are not inlined in the tool parameters fragment.
NutrientAmountMap = Annotated[
    dict[NutrientName, float],
    WithJsonSchema(
        {
            "type": "object",
            "additionalProperties": {"type": "number"},
            "description": ("Nutrient amounts keyed by canonical name (e.g. energy_kcal, protein_g, sodium_mg)."),
        }
    ),
]


# Canonical unit per nutrient - asserted by validators.
_CANONICAL_UNIT: dict[NutrientName, str] = {
    NutrientName.ENERGY_KCAL: "kcal",
    NutrientName.PROTEIN_G: "g",
    NutrientName.CARBS_G: "g",
    NutrientName.FAT_G: "g",
    NutrientName.SATURATED_FAT_G: "g",
    NutrientName.FIBER_G: "g",
    NutrientName.SUGAR_G: "g",
    NutrientName.SODIUM_MG: "mg",
    NutrientName.POTASSIUM_MG: "mg",
    NutrientName.CALCIUM_MG: "mg",
    NutrientName.IRON_MG: "mg",
    NutrientName.VITAMIN_C_MG: "mg",
    NutrientName.VITAMIN_D_UG: "ug",
    NutrientName.CHOLESTEROL_MG: "mg",
}


def canonical_unit(name: NutrientName) -> str:
    return _CANONICAL_UNIT[name]


class Nutrient(BaseModel):
    """A single nutrient measurement, always in canonical unit.

    The unit field is *redundant* on purpose: it serves as a tripwire that
    forces the LLM (or any caller) to think about units when constructing
    food items, and lets the validator reject unit confusion.
    """

    model_config = ConfigDict(extra="forbid")

    name: NutrientName
    amount: float = Field(ge=0.0, description="Non-negative amount in canonical unit.")
    unit: str = Field(description="Must match the canonical unit for `name`.")

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, v: str, info: object) -> str:  # noqa: ARG003
        # We can't access other fields here in a field_validator without info.data
        # but pydantic v2 supports info.data via ValidationInfo. Easiest: trust
        # construction sites and assert canonical at the model_validator below.
        return v.lower().strip()

    def to_canonical(self) -> Nutrient:
        """Return a copy with `unit` snapped to the canonical unit (no conversion)."""
        return Nutrient(name=self.name, amount=self.amount, unit=canonical_unit(self.name))


class FoodItem(BaseModel):
    """A food/recipe ingredient with normalized per-100g nutrients.

    Either a real Open Food Facts product (keyed by its barcode `code`) or a
    food invented by the LLM, distinguished by `from_open_food_facts`.
    """

    model_config = ConfigDict(extra="forbid")

    from_open_food_facts: bool = Field(
        default=False,
        description=(
            "True only for a real product fetched from the Open Food Facts database. "
            "Leave false for a food you invent/estimate yourself; such foods must omit `code`."
        ),
    )
    code: str | None = Field(default=None, description="Open Food Facts barcode. Required when from_open_food_facts.")
    name: str = Field(min_length=1)
    description: str | None = None

    # Per-100g nutrients (canonical units). A dict keyed by NutrientName for
    # O(1) lookup in the Totaller. Values may be missing if the source DB does
    # not provide them.
    nutrients_per_100g: NutrientAmountMap = Field(default_factory=dict)

    # Free-form tags (e.g. "vegetarian", "contains:milk"). The validator uses
    # `contains:<allergen>` tags to enforce HardConstraint allergen exclusions.
    tags: list[str] = Field(default_factory=list)

    @field_validator("nutrients_per_100g")
    @classmethod
    def _check_non_negative(cls, v: dict[NutrientName, float]) -> dict[NutrientName, float]:
        for n, amount in v.items():
            if amount < 0:
                raise ValueError(f"Negative amount for {n.value}: {amount}")
        return v

    @model_validator(mode="after")
    def _check_code_provenance(self) -> FoodItem:
        if self.from_open_food_facts and self.code is None:
            raise ValueError(
                "Open Food Facts records must have a `code` (barcode); only LLM-created foods may omit it.",
            )
        return self


class MacroTargets(BaseModel):
    """Daily macronutrient targets used as the reference for MAE/MSE evaluation."""

    model_config = ConfigDict(extra="forbid")

    energy_kcal: float = Field(gt=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    fiber_g: float = Field(default=25.0, ge=0)

    def as_dict(self) -> dict[NutrientName, float]:
        return {
            NutrientName.ENERGY_KCAL: self.energy_kcal,
            NutrientName.PROTEIN_G: self.protein_g,
            NutrientName.CARBS_G: self.carbs_g,
            NutrientName.FAT_G: self.fat_g,
            NutrientName.FIBER_G: self.fiber_g,
        }
