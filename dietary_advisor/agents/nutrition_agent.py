"""Nutrition agent: owns the Totaller and food-database lookup tools.

This is the workhorse agent in the pipeline. It is fed a profile and macro
targets and is expected to return a `MealPlan` (structured output) that
respects the profile (allergens, diet pattern) and approximates the targets.
It never sees hard constraints - those are an evaluation-only concept.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.prompts import NUTRITION_AGENT_SYSTEM, REFLECTION_REFINER_SYSTEM
from dietary_advisor.config import get_settings
from dietary_advisor.llm import resolve_llm_model
from dietary_advisor.schemas.meal_plan import MealPlan
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName
from dietary_advisor.tools.totaller import total_meal_plan_dict

log = logging.getLogger(__name__)


def build_nutrition_agent(
    model: str | Model | None = None,
    *,
    totaller_enabled: bool = True,
) -> Agent[AgentDeps, MealPlan]:
    """Construct a fresh `Agent` instance bound to AgentDeps + MealPlan output."""
    settings = get_settings()
    agent: Agent[AgentDeps, MealPlan] = Agent(
        resolve_llm_model(model or settings.llm_model),
        deps_type=AgentDeps,
        output_type=MealPlan,
        system_prompt=NUTRITION_AGENT_SYSTEM,
        retries=2,
    )

    def _hits_summary(items: list[FoodItem]) -> list[dict[str, Any]]:
        return [
            {
                "code": it.code,
                "name": it.name,
                "tags": it.tags,
                "kcal_per_100g": it.nutrients_per_100g.get(NutrientName.ENERGY_KCAL, 0.0)
                if isinstance(it.nutrients_per_100g, dict)
                else 0.0,
            }
            for it in items
        ]

    @agent.tool
    async def lookup_food(ctx: RunContext[AgentDeps], query: str, max_results: int = 2) -> list[dict[str, Any]]:
        """Search the food database by name and return the best-matching products.

        Each hit includes its `code` (barcode) and per-100g calories so the LLM
        can pick foods and portion them. Returns an empty list when nothing
        matches - fall back to your own nutritional knowledge in that case.
        """
        if ctx.deps.food_db is None:
            return []
        try:
            items = ctx.deps.food_db.search(query, limit=max_results)
        except Exception as exc:  # noqa: BLE001
            log.warning("Food DB search failed for %r: %s", query, exc)
            return []
        return _hits_summary(items)

    @agent.tool
    async def lookup_foods(
        ctx: RunContext[AgentDeps],
        queries: list[str],
        max_results_per_query: int = 2,
    ) -> list[dict[str, Any]]:
        """Batch food-database search: one tool call for several ingredient queries."""
        if ctx.deps.food_db is None or not queries:
            return []
        hits: list[FoodItem] = []
        for query in queries[:6]:
            try:
                hits.extend(ctx.deps.food_db.search(query, limit=max_results_per_query))
            except Exception as exc:  # noqa: BLE001
                log.warning("Food DB search failed for %r: %s", query, exc)
        return _hits_summary(hits)

    if totaller_enabled:

        @agent.tool
        async def total_meal_plan(ctx: RunContext[AgentDeps], plan: MealPlan) -> dict[str, float]:  # noqa: ARG001
            """Deterministically sum the per-nutrient totals for a draft `MealPlan`."""
            return total_meal_plan_dict(plan)

    return agent


def build_refiner_agent(
    model: str | Model | None = None,
    *,
    totaller_enabled: bool = True,
) -> Agent[AgentDeps, MealPlan]:
    """Self-review agent for the Reflection Loop.

    Same toolset as the main nutrition agent, but a prompt focused on
    critiquing and improving the previous plan (no deterministic feedback -
    the loop is a plain review/correction pass).
    """
    settings = get_settings()
    agent: Agent[AgentDeps, MealPlan] = Agent(
        resolve_llm_model(model or settings.llm_model),
        deps_type=AgentDeps,
        output_type=MealPlan,
        system_prompt=REFLECTION_REFINER_SYSTEM,
        retries=2,
    )

    if totaller_enabled:

        @agent.tool
        async def total_meal_plan(ctx: RunContext[AgentDeps], plan: MealPlan) -> dict[str, float]:  # noqa: ARG001
            return total_meal_plan_dict(plan)

    return agent


def format_self_review_prompt(plan: MealPlan) -> str:
    """Render a plain self-review prompt for the refiner agent (no deterministic feedback)."""
    return (
        "Review the following MealPlan critically: check it against the user's profile "
        "(allergens, diet pattern, disliked foods), verify the macro totals with "
        "`total_meal_plan`, and fix any issues you find.\n\n"
        "Previous MealPlan (JSON):\n"
        f"{plan.model_dump_json(indent=2)}\n\n"
        "Produce an improved MealPlan."
    )
