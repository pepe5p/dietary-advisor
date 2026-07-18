"""Re-exports of the production hydration step for evaluation callers.

Hydration (`AgentMealPlan` -> `MealPlan`) moved to `dietary_advisor.hydration`
so the production pipeline resolves the same way the evaluation harness
always has; this module just keeps the existing `evaluation.validation.*`
import surface working.
"""

from __future__ import annotations

from dietary_advisor.hydration import hydrate_meal_plan, to_food_item, total_agent_meal_plan

__all__ = [
    "hydrate_meal_plan",
    "to_food_item",
    "total_agent_meal_plan",
]
