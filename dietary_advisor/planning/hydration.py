"""Hydrate agent-emitted references into full domain entities.

The nutrition agent never sees a `FoodItem`; it only ever sees search-result
summaries and returns `PortionRef` codes (see `agents.agent_output`). This
is the one place those codes resolve back to a `FoodItem`/`MealPlan`, both in
production (`Pipeline.run`) and in the evaluation harness, so the LLM can
never fake a nutrient value.
"""

from __future__ import annotations

import logging

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.food_db import FoodDb, OFFItem, USDAItem
from dietary_advisor.planning.meal_plan import Meal, MealPlan, NutrientTotals, Portion
from dietary_advisor.totaller.aggregate import total_meal_plan
from dietary_advisor.totaller.nutrition import FoodItem

log = logging.getLogger(__name__)


def to_food_item(item: OFFItem | USDAItem) -> FoodItem:
    """Map a food_db read model to the domain `FoodItem` (Totaller/shopping-list currency)."""
    return FoodItem(
        code=item.code,
        name=item.name,
        description=item.description,
        nutrients_per_100g=item.nutrients_per_100g,
        tags=item.tags if isinstance(item, OFFItem) else [],
    )


def hydrate_meal_plan(plan: AgentMealPlan, food_db: FoodDb) -> MealPlan:
    """Resolve every `PortionRef` to a full `FoodItem` and build a `MealPlan`."""
    meals: list[Meal] = []
    for agent_meal in plan.meals:
        portions: list[Portion] = []
        for ref in agent_meal.recipe.portions:
            item = food_db.get_food(ref.code)
            portions.append(Portion(food=to_food_item(item), grams=ref.grams))
        meals.append(
            Meal(
                kind=agent_meal.kind,
                name=agent_meal.recipe.name,
                portions=portions,
                recipe=agent_meal.recipe.instructions,
            ),
        )
    log.debug("hydration.hydrate_meal_plan(%d meal(s))", len(meals))
    return MealPlan(
        user_id=plan.user_id,
        meals=meals,
        rationale=plan.rationale,
        citations=plan.citations,
    )


def total_agent_meal_plan(plan: AgentMealPlan, food_db: FoodDb) -> NutrientTotals:
    """Sum nutrients for an agent-emitted plan, resolving `code` via `food_db`."""
    return total_meal_plan(hydrate_meal_plan(plan, food_db))
