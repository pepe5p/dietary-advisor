"""MealPlan schemas: the structured output the LLM must produce.

By forcing the LLM to commit to a Pydantic-typed `MealPlan` (via pydantic-ai
structured output), we obtain a deterministic surface that the symbolic
Totaller and Validator can then operate on without ambiguity.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.totaller.nutrition import FoodItem, NutrientAmountMap, NutrientName


class Portion(BaseModel):
    """A precise portion of a single food item, in grams."""

    model_config = ConfigDict(extra="forbid")

    food: FoodItem
    grams: float = Field(gt=0, description="Edible mass in grams.")


class Meal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str = Field(min_length=1)
    portions: list[Portion] = Field(min_length=1)
    recipe: str = Field(
        min_length=120,
        description=(
            "Full step-by-step preparation method (prep, cook method/temperature/"
            "time, assembly) - detailed enough to cook from without any other reference."
        ),
    )


class Citation(BaseModel):
    """A pointer back to the RAG corpus chunk that grounds a rationale claim."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Document identifier, e.g. 'NICE_NG28'.")
    page: int | None = None
    snippet: str = Field(min_length=1, description="Verbatim quote (<=300 chars).")


class MealNutrientTotals(BaseModel):
    """Per-nutrient totals for a single meal within a `NutrientTotals` breakdown."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    totals: NutrientAmountMap = Field(default_factory=dict)

    def get(self, name: NutrientName, default: float = 0.0) -> float:
        return self.totals.get(name, default)


class NutrientTotals(BaseModel):
    """Result of the Totaller run over a `MealPlan`. All values in canonical units."""

    model_config = ConfigDict(extra="forbid")

    totals: NutrientAmountMap = Field(default_factory=dict)
    # Ordered like MealPlan.meals; a list (not a dict) because meal names can collide.
    per_meal: list[MealNutrientTotals] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Per-nutrient data-coverage caveats: which totals are understated "
            "because some foods have no value for that nutrient in the source DB."
        ),
    )

    def get(self, name: NutrientName, default: float = 0.0) -> float:
        return self.totals.get(name, default)


class MealPlan(BaseModel):
    """Top-level structured artifact produced by the agentic layer."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    meals: list[Meal] = Field(min_length=1)
    rationale: str = Field(
        default="",
        description=(
            "Patient-facing explanation of the day's plan, plus any supplementation "
            "the profile warrants (nutrient, reason, form and dose range); each "
            "clinical claim should be backed by a citation when available."
        ),
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="RAG citations grounding `rationale` in the clinical-guideline corpus.",
    )


class ShoppingListItem(BaseModel):
    """One consolidated ingredient line: total mass needed across the whole plan."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    total_grams: float = Field(gt=0)
    code: str | None = None
    # Deterministically derived from `total_grams` and the ingredient's
    # per-100g nutrients (same Fraction-based math as the Totaller), so the
    # shopping list can be read as a standalone macro summary per ingredient.
    energy_kcal: float = Field(default=0.0, ge=0.0)
    protein_g: float = Field(default=0.0, ge=0.0)
    carbs_g: float = Field(default=0.0, ge=0.0)
    fat_g: float = Field(default=0.0, ge=0.0)
    other_nutrients: NutrientAmountMap = Field(default_factory=dict)
    quantity_g: float | None = None


class ShoppingList(BaseModel):
    """Deterministically derived grocery list for a `MealPlan`."""

    model_config = ConfigDict(extra="forbid")

    items: list[ShoppingListItem] = Field(default_factory=list)
