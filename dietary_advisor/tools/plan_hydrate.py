"""Hydrate :class:`~dietary_advisor.schemas.agent_output.AgentMealPlan` via cache lookup."""

from __future__ import annotations

from dietary_advisor.schemas.agent_output import AgentMealPlan
from dietary_advisor.schemas.meal_plan import Meal, MealPlan, Portion, Recipe
from dietary_advisor.tools.food_lookup import FoodLookup


def hydrate_meal_plan(plan: AgentMealPlan, lookup: FoodLookup) -> MealPlan:
    """Resolve every ``PortionRef`` to a full ``FoodItem`` and build a ``MealPlan``."""
    meals: list[Meal] = []
    for agent_meal in plan.meals:
        portions: list[Portion] = []
        for ref in agent_meal.recipe.portions:
            food = lookup.get_food(ref.fdc_id)
            portions.append(Portion(food=food, grams=ref.grams))
        meals.append(
            Meal(
                kind=agent_meal.kind,
                recipe=Recipe(
                    name=agent_meal.recipe.name,
                    portions=portions,
                    instructions=agent_meal.recipe.instructions,
                ),
            ),
        )
    return MealPlan(
        user_id=plan.user_id,
        meals=meals,
        rationale=plan.rationale,
        citations=plan.citations,
    )
