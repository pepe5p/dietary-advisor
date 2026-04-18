"""MealPlan schemas: the structured output the LLM must produce.

By forcing the LLM to commit to a Pydantic-typed `MealPlan` (via pydantic-ai
structured output), we obtain a deterministic surface that the symbolic
Totaller and Validator can then operate on without ambiguity.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.schemas.nutrition import FoodItem, NutrientName


class MealKind(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class Portion(BaseModel):
    """A precise portion of a single food item, in grams."""

    model_config = ConfigDict(extra="forbid")

    food: FoodItem
    grams: float = Field(gt=0, description="Edible mass in grams.")


class Recipe(BaseModel):
    """A simple recipe = ordered list of weighted portions plus optional notes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    portions: list[Portion] = Field(min_length=1)
    instructions: str | None = None


class Meal(BaseModel):
    """A meal slot containing one recipe."""

    model_config = ConfigDict(extra="forbid")

    kind: MealKind
    recipe: Recipe


class Citation(BaseModel):
    """A pointer back to the RAG corpus chunk that grounds a rationale claim."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Document identifier, e.g. 'NICE_NG28'.")
    section: str | None = None
    page: int | None = None
    snippet: str = Field(min_length=1, description="Verbatim quote (<=300 chars).")


class NutrientTotals(BaseModel):
    """Result of the Totaller run over a `MealPlan`. All values in canonical units."""

    model_config = ConfigDict(extra="forbid")

    totals: dict[NutrientName, float] = Field(default_factory=dict)

    def get(self, name: NutrientName, default: float = 0.0) -> float:
        return self.totals.get(name, default)


class MealPlan(BaseModel):
    """Top-level structured artifact produced by the agentic layer."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    meals: list[Meal] = Field(min_length=1)
    rationale: str = Field(
        default="",
        description="Patient-facing explanation; each claim should be backed by a citation.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="RAG citations supporting `rationale`. Used to compute Faithfulness.",
    )
