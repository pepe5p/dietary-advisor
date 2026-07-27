"""Meal-idea schema: creative dish concepts, not DB-verified facts.

`MealConcept` is the output of the meal-idea agent (see
`dietary_advisor.agents.meal_idea`), a brainstorming pass that runs
before the nutrition agent ever touches the food DB. It intentionally carries
no nutrients or product codes - it exists purely to give the nutrition agent
a concrete, varied creative starting point instead of an abstract macro gap
to fill, and never contributes a `PortionRef` itself.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MealConcept(BaseModel):
    """One brainstormed dish concept.

    Several concepts share a `kind`: the agent proposes a few alternatives per
    meal slot and the nutrition agent builds one meal from each slot's options.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="Meal slot, e.g. breakfast, lunch, dinner, snack.")
    dish_name: str = Field(
        min_length=1,
        description="A concrete, specific dish name - not a generic nutrient-role label.",
    )
