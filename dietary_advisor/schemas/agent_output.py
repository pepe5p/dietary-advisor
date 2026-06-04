"""Evaluation contract: meal plans reference USDA cache IDs, not embedded nutrients.

The production nutrition agent still returns :class:`~dietary_advisor.schemas.meal_plan.MealPlan`
with full :class:`~dietary_advisor.schemas.nutrition.FoodItem` payloads. The ablation harness
hydrates :class:`AgentMealPlan` via :class:`~dietary_advisor.tools.food_lookup.FoodLookup`
before running the Totaller and structural Validator.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.schemas.meal_plan import Citation, MealKind


class PortionRef(BaseModel):
    """A portion keyed by cached USDA ``fdc_id`` and mass in grams."""

    model_config = ConfigDict(extra="forbid")

    fdc_id: int = Field(gt=0, description="USDA FoodData Central ID from the prefetch cache.")
    grams: float = Field(gt=0, description="Edible mass in grams.")


class AgentRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    portions: list[PortionRef] = Field(min_length=1)
    instructions: str | None = None


class AgentMeal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MealKind
    recipe: AgentRecipe


class AgentMealPlan(BaseModel):
    """Structured plan for evaluation: ingredients by ``fdc_id``, no aggregate macros."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    meals: list[AgentMeal] = Field(default_factory=list)
    rationale: str = Field(default="")
    citations: list[Citation] = Field(default_factory=list)
