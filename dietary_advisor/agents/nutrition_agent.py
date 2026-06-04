"""Nutrition agent: owns the Totaller, USDA lookup, and MILP optimisation tools.

This is the workhorse agent in the pipeline. It is fed a profile + targets +
constraints and is expected to return a `MealPlan` (structured output) that
respects the constraints and approximates the macro targets.
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
from dietary_advisor.schemas.constraints import ValidationReport
from dietary_advisor.schemas.meal_plan import MealPlan
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName
from dietary_advisor.tools.milp_optimizer import OptimisationResult, optimize_portions
from dietary_advisor.tools.totaller import total_meal_plan_dict

log = logging.getLogger(__name__)


def build_nutrition_agent(model: str | Model | None = None) -> Agent[AgentDeps, MealPlan]:
    """Construct a fresh `Agent` instance bound to AgentDeps + MealPlan output."""
    settings = get_settings()
    agent: Agent[AgentDeps, MealPlan] = Agent(
        resolve_llm_model(model or settings.llm_model),
        deps_type=AgentDeps,
        output_type=MealPlan,
        system_prompt=NUTRITION_AGENT_SYSTEM,
        retries=2,
    )

    def _append_usda_hits(ctx: RunContext[AgentDeps], items: list[FoodItem]) -> list[dict[str, Any]]:
        ctx.deps.shortlist.extend(items)
        return [
            {
                "fdc_id": it.fdc_id,
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
        """Search USDA FoodData Central and add the top results to the shortlist.

        Returns a JSON-friendly summary so the LLM can make sensible portioning
        decisions; the full `FoodItem` objects are kept in `ctx.deps.shortlist`.
        """
        if ctx.deps.usda is None:
            return []
        try:
            items = ctx.deps.usda.search(query, page_size=max_results)
        except Exception as exc:  # noqa: BLE001
            log.warning("USDA search failed for %r: %s", query, exc)
            return []
        return _append_usda_hits(ctx, items)

    @agent.tool
    async def lookup_foods(
        ctx: RunContext[AgentDeps],
        queries: list[str],
        max_results_per_query: int = 2,
    ) -> list[dict[str, Any]]:
        """Batch USDA search: one tool call for several ingredient queries."""
        if ctx.deps.usda is None or not queries:
            return []
        hits: list[FoodItem] = []
        for query in queries[:6]:
            try:
                hits.extend(ctx.deps.usda.search(query, page_size=max_results_per_query))
            except Exception as exc:  # noqa: BLE001
                log.warning("USDA search failed for %r: %s", query, exc)
        return _append_usda_hits(ctx, hits)

    @agent.tool
    async def total_meal_plan(ctx: RunContext[AgentDeps], plan: MealPlan) -> dict[str, float]:  # noqa: ARG001
        """Deterministically sum the per-nutrient totals for a draft `MealPlan`."""
        return total_meal_plan_dict(plan)

    @agent.tool
    async def optimize_portions_tool(
        ctx: RunContext[AgentDeps],
        food_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the MILP solver over the current shortlist and return gram amounts.

        If `food_names` is supplied, restrict the candidate set to those names
        (case-insensitive). Otherwise use the entire shortlist.
        """
        candidates: list[FoodItem]
        if food_names:
            wanted = {n.lower().strip() for n in food_names}
            candidates = [f for f in ctx.deps.shortlist if f.name.lower() in wanted]
        else:
            candidates = list(ctx.deps.shortlist)
        if not candidates:
            return {"status": "NoEligibleFoods", "grams": {}, "totals": {}}
        result: OptimisationResult = optimize_portions(
            candidates,
            ctx.deps.targets,
            ctx.deps.constraints,
        )
        return {
            "status": result.status,
            "objective": result.objective,
            "grams": result.grams,
            "totals": {k.value: v for k, v in result.totals.items()},
        }

    return agent


def build_refiner_agent(model: str | Model | None = None) -> Agent[AgentDeps, MealPlan]:
    """Critique-and-refine agent for the Reflection Loop.

    Same toolset as the main nutrition agent, but a more targeted prompt that
    focuses on fixing specific reported violations.
    """
    settings = get_settings()
    agent: Agent[AgentDeps, MealPlan] = Agent(
        resolve_llm_model(model or settings.llm_model),
        deps_type=AgentDeps,
        output_type=MealPlan,
        system_prompt=REFLECTION_REFINER_SYSTEM,
        retries=2,
    )

    @agent.tool
    async def total_meal_plan(ctx: RunContext[AgentDeps], plan: MealPlan) -> dict[str, float]:  # noqa: ARG001
        return total_meal_plan_dict(plan)

    return agent


def format_violations_prompt(plan: MealPlan, report: ValidationReport) -> str:
    """Render a violation list as a refinement prompt for the refiner agent."""
    bullets = "\n".join(
        f"- {v.detail} (constraint: {v.constraint.kind}={v.constraint.target})" for v in report.violations
    )
    return (
        "Previous MealPlan failed validation with the following hard violations:\n"
        f"{bullets}\n\n"
        "Previous MealPlan (JSON):\n"
        f"{plan.model_dump_json(indent=2)}\n\n"
        "Produce a corrected MealPlan that addresses every violation."
    )
