"""Hydrate :class:`~dietary_advisor.schemas.agent_output.AgentMealPlan` via cache lookup."""

from __future__ import annotations

import logging

from dietary_advisor.schemas.agent_output import AgentMealPlan
from dietary_advisor.schemas.meal_plan import Meal, MealPlan, Portion, Recipe
from dietary_advisor.tools.food_db import OffFoodDb

log = logging.getLogger(__name__)


def hydrate_meal_plan(plan: AgentMealPlan, lookup: OffFoodDb) -> MealPlan:
    """Resolve every ``PortionRef`` to a full ``FoodItem`` and build a ``MealPlan``."""
    meals: list[Meal] = []
    for agent_meal in plan.meals:
        portions: list[Portion] = []
        for ref in agent_meal.recipe.portions:
            food = lookup.get_food(ref.code)
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
    log.info("plan_hydrate.hydrate_meal_plan(%d meal(s))", len(meals))
    return MealPlan(
        user_id=plan.user_id,
        meals=meals,
        rationale=plan.rationale,
        citations=plan.citations,
    )
