"""Agent output contract: the structured plan the nutrition agent must produce.

`PortionRef` deliberately carries no nutrients - the agent has no ability to
invent or estimate them. It can only reference a `code` (and matching `name`)
copied verbatim from a `lookup_foods` result; an output
validator on the agent rejects any other code. `dietary_advisor.planning.hydration`
is the one place a reference resolves back to a real, DB-verified
`FoodItem`, both in production (`Pipeline.run`) and in the evaluation
harness.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.planning.meal_plan import Citation


class PortionRef(BaseModel):
    """A portion keyed by a verified food-database code and mass in grams."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        description=("Code copied verbatim from a lookup result: an `off:<barcode>` or `usda:<fdc_id>` id."),
    )
    name: str = Field(
        min_length=1,
        description="Product name copied from that same lookup result (grounding/readability only).",
    )
    grams: float = Field(gt=0, description="Edible mass in grams.")


class AgentRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    portions: list[PortionRef] = Field(min_length=1)
    instructions: str = Field(
        min_length=120,
        description=(
            "Full step-by-step preparation method (prep, cook method/temperature/"
            "time, assembly) - detailed enough to cook from without any other reference."
        ),
    )


class AgentMeal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    recipe: AgentRecipe


class AgentMealPlan(BaseModel):
    """Structured output the nutrition agent must produce: ingredients by database code, no embedded nutrients."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    meals: list[AgentMeal] = Field(min_length=1)
    rationale: str = Field(
        min_length=200,
        description=(
            "Patient-facing explanation of the day's plan, plus any supplementation "
            "the profile warrants (nutrient, reason, form and dose range)."
        ),
    )
    citations: list[Citation] = Field(default_factory=list)
