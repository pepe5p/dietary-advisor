"""Nutrition agent: owns the Totaller and food-database lookup tools.

This is the workhorse agent in the pipeline. It is fed a profile and macro
targets and is expected to return an `AgentMealPlan` (structured output) that
respects the profile (allergens, diet pattern) and approximates the targets.
It never sees hard constraints - those are an evaluation-only concept.

The agent cannot invent a food or its nutrients: every `PortionRef.code` must
come from a `lookup_food`/`lookup_foods` hit, and `_validate_codes` rejects
any output whose codes don't resolve against the real database, forcing a
retry rather than letting a fabricated code reach hydration.
"""

from __future__ import annotations

import logging

from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.prompts import nutrition_agent_system, REFLECTION_REFINER_AGENT_SYSTEM
from dietary_advisor.config import get_settings
from dietary_advisor.food_db.errors import UnknownFoodCodeError
from dietary_advisor.food_db.facade import LookupQuery, LookupResult
from dietary_advisor.planning.hydration import total_agent_meal_plan
from dietary_advisor.planning.meal_plan import NutrientTotals

log = logging.getLogger(__name__)


async def _validate_codes(ctx: RunContext[AgentDeps], output: AgentMealPlan) -> AgentMealPlan:
    """Reject any portion whose `code` doesn't resolve against the real food DB.

    The mechanical backstop for "the agent cannot invent a food": a code that
    isn't a genuine lookup result is caught here and sent back as a retry,
    rather than silently reaching hydration.
    """
    unknown: list[str] = []
    for meal in output.meals:
        for ref in meal.recipe.portions:
            try:
                ctx.deps.food_db.get_food(ref.code)
            except UnknownFoodCodeError:
                unknown.append(ref.code)
    if unknown:
        raise ModelRetry(
            f"These codes do not exist in the food database: {sorted(set(unknown))}. "
            "Every `code` must be copied verbatim from a `lookup_food`/`lookup_foods` result - "
            "search again and use a code you actually received."
        )
    return output


async def lookup_food(
    ctx: RunContext[AgentDeps],
    query: str,
    max_results_usda: int = 2,
    max_results_off: int = 2,
) -> LookupResult:
    """Search both food databases by name for the best-matching foods.

    Results come back grouped by source: `open_food_facts` (branded/packaged
    products, often Polish) and `usda` (generic whole foods and reference
    ingredients like raw carrot or plain chicken breast). Each hit carries a
    verified `code`, `name`, and per-100g macros - copy both `code` and `name`
    verbatim into the `PortionRef` for that ingredient. You cannot use any
    food that isn't a hit from this tool.

    `max_results_usda`/`max_results_off` cap hits per source independently
    (0 skips that source) - weight towards whichever source actually stocks
    this ingredient.
    """
    log.info("lookup_food(%r, max_results_usda=%d, max_results_off=%d)", query, max_results_usda, max_results_off)
    lookup_query = LookupQuery(query=query, max_results_off=max_results_off, max_results_usda=max_results_usda)
    return await ctx.deps.food_db.lookup([lookup_query])


async def lookup_foods(
    ctx: RunContext[AgentDeps],
    queries: list[LookupQuery],
) -> LookupResult:
    """Batch search of both food databases: one tool call for several ingredient queries.

    Each query sets its own `max_results_usda`/`max_results_off`, so you can
    weight some ingredients towards USDA, others towards OFF, or skip a
    source entirely. Returns hits grouped by source (`open_food_facts`,
    `usda`) exactly like `lookup_food`, pooling matches across every query.
    """
    log.info("lookup_foods(%d query/queries)", len(queries))
    if not queries:
        return LookupResult()
    return await ctx.deps.food_db.lookup(queries)


async def total_meal_plan(ctx: RunContext[AgentDeps], plan: AgentMealPlan) -> NutrientTotals:
    """Deterministically hydrate `plan`'s codes and total nutrients, overall and per meal."""
    try:
        return total_agent_meal_plan(plan, ctx.deps.food_db)
    except UnknownFoodCodeError as exc:
        raise ModelRetry(
            f"Cannot total the plan: {exc}. Every `code` must be copied verbatim from a lookup result.",
        ) from exc


def _build_agent(system_prompt: str, *, totaller_enabled: bool) -> Agent[AgentDeps, AgentMealPlan]:
    settings = get_settings()
    agent: Agent[AgentDeps, AgentMealPlan] = Agent(
        settings.resolved_llm_model,
        deps_type=AgentDeps,
        output_type=AgentMealPlan,
        system_prompt=system_prompt,
        model_settings=ModelSettings(temperature=settings.llm_temperature),
        retries=2,
    )
    agent.output_validator(_validate_codes)
    agent.tool(lookup_food)
    agent.tool(lookup_foods)
    if totaller_enabled:
        agent.tool(total_meal_plan)
    return agent


def build_nutrition_agent(
    *,
    totaller_enabled: bool = True,
    rag_enabled: bool = True,
) -> Agent[AgentDeps, AgentMealPlan]:
    """Construct a fresh `Agent` instance bound to AgentDeps + AgentMealPlan output.

    `rag_enabled` only shapes the prompt (whether the citation rule is stated);
    the nutrition agent never owns the retriever as a tool.
    """
    prompt = nutrition_agent_system(totaller_enabled=totaller_enabled, rag_enabled=rag_enabled)
    return _build_agent(prompt, totaller_enabled=totaller_enabled)


def build_refiner_agent(*, totaller_enabled: bool = True) -> Agent[AgentDeps, AgentMealPlan]:
    """Refinement agent for the Reflection Loop.

    Same toolset as the main nutrition agent (including the lookup tools, so
    it can swap out an ingredient rather than inventing a replacement), but a
    prompt focused on fixing exactly the issues the critic agent reported
    (see `dietary_advisor.reflection`), not a blind self-review pass.
    """
    return _build_agent(REFLECTION_REFINER_AGENT_SYSTEM, totaller_enabled=totaller_enabled)
