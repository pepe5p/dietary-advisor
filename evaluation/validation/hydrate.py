"""Adapt pipeline ``MealPlan`` output to the evaluation contract."""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.schemas.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.schemas.meal_plan import MealPlan
from dietary_advisor.tools.plan_hydrate import hydrate_meal_plan as _hydrate_meal_plan

# Re-export for evaluation package consumers.
hydrate_meal_plan = _hydrate_meal_plan


@dataclass(frozen=True)
class EvalPlanConversion:
    """Result of converting a pipeline ``MealPlan`` to the evaluation contract."""

    plan: AgentMealPlan
    warnings: tuple[str, ...]


def meal_plan_to_eval_plan(meal_plan: MealPlan) -> EvalPlanConversion:
    """Extract ``fdc_id`` + ``grams`` from a pipeline plan for evaluation metrics."""
    warnings: list[str] = []
    agent_meals: list[AgentMeal] = []
    for meal in meal_plan.meals:
        refs: list[PortionRef] = []
        for portion in meal.recipe.portions:
            fdc_id = portion.food.fdc_id
            if fdc_id is None:
                warnings.append(
                    f"portion in {meal.kind.value}/{meal.recipe.name!r} missing fdc_id ({portion.food.name!r})",
                )
                continue
            refs.append(PortionRef(fdc_id=fdc_id, grams=portion.grams))
        if not refs:
            warnings.append(f"meal {meal.kind.value}/{meal.recipe.name!r} has no portions with fdc_id")
            continue
        agent_meals.append(
            AgentMeal(
                kind=meal.kind,
                recipe=AgentRecipe(
                    name=meal.recipe.name,
                    portions=refs,
                    instructions=meal.recipe.instructions,
                ),
            ),
        )
    if not agent_meals:
        warnings.append("no meals with resolvable fdc_id portions")
    return EvalPlanConversion(
        plan=AgentMealPlan(
            user_id=meal_plan.user_id,
            meals=agent_meals,
            rationale=meal_plan.rationale,
            citations=meal_plan.citations,
        ),
        warnings=tuple(warnings),
    )
