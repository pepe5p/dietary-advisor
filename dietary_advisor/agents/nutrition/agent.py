"""Nutrition agent: owns the Totaller and food-database lookup tools.

This is the workhorse agent in the pipeline. It is fed a profile and macro
targets and is expected to return an `AgentMealPlan` (structured output) that
respects the profile (allergens, diet pattern) and approximates the targets.
It never sees hard constraints - those are an evaluation-only concept.

The agent cannot invent a food or its nutrients: every `PortionRef.code` must
come from a `lookup_foods` hit, and `_validate_codes` rejects
any output whose codes don't resolve against the real database, forcing a
retry rather than letting a fabricated code reach hydration.
"""

from __future__ import annotations

import logging

from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext
from pydantic_ai.models import Model

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition.prompts import nutrition_agent_system, REFLECTION_REFINER_AGENT_SYSTEM
from dietary_advisor.config import get_settings
from dietary_advisor.food_db.errors import UnknownFoodCodeError
from dietary_advisor.food_db.facade import BatchLookupResult, LookupQuery
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
            "Every `code` must be copied verbatim from a `lookup_foods` result - "
            "search again and use a code you actually received."
        )
    return output


async def lookup_foods(
    ctx: RunContext[AgentDeps],
    queries: list[LookupQuery],
) -> str:
    """Batch search of both food databases: one tool call for several ingredient queries.

    Returns compact JSON with one entry per input query in input order.
    Each entry has `query`, then `results.open_food_facts` (branded/packaged
    products, often Polish) and `results.usda` (generic whole foods and
    reference ingredients like raw carrot or plain chicken breast). Each hit
    object carries a verified `code`, a `name`, context fields, and per-100g
    nutrients in canonical units (`energy_kcal`, `protein_g`, `sodium_mg`, …);
    unknown nutrients are omitted. Copy `code` and `name` verbatim into the
    `PortionRef` for that ingredient - `code` holds the identifier alone, never
    the name or grams appended to it. You cannot use any food that isn't a hit
    from this tool.

    Each query sets its own `max_results_usda`/`max_results_off`, so you can
    weight some ingredients towards USDA, others towards OFF, or skip a source
    entirely (0 skips that source). Pass a single-element list for a one-off
    follow-up search.
    """
    log.info("lookup_foods(%d query/queries)", len(queries))
    if not queries:
        return BatchLookupResult().render_json()
    result = await ctx.deps.food_db.lookup(queries)
    return result.render_json()


async def total_meal_plan(ctx: RunContext[AgentDeps], plan: AgentMealPlan) -> NutrientTotals:
    """Deterministically hydrate `plan`'s codes and total nutrients, overall and per meal.

    Check ``warnings`` for nutrients where some foods had no DB value: those totals
    are a floor, not a complete measurement.
    """
    try:
        return total_agent_meal_plan(plan, ctx.deps.food_db)
    except UnknownFoodCodeError as exc:
        raise ModelRetry(
            f"Cannot total the plan: {exc}. Every `code` must be copied verbatim from a lookup result.",
        ) from exc


def _build_agent(
    system_prompt: str,
    *,
    totaller_enabled: bool,
    model: Model | None = None,
) -> Agent[AgentDeps, AgentMealPlan]:
    settings = get_settings()
    agent: Agent[AgentDeps, AgentMealPlan] = Agent(
        model if model is not None else settings.resolved_llm_model,
        deps_type=AgentDeps,
        output_type=AgentMealPlan,
        system_prompt=system_prompt,
        model_settings=ModelSettings(temperature=settings.llm_temperature),
        retries=2,
    )
    agent.output_validator(_validate_codes)
    agent.tool(lookup_foods)
    if totaller_enabled:
        agent.tool(total_meal_plan)
    return agent


def build_nutrition_agent(
    *,
    totaller_enabled: bool = True,
    rag_enabled: bool = True,
    has_user_request: bool = True,
    model: Model | None = None,
) -> Agent[AgentDeps, AgentMealPlan]:
    """Construct a fresh `Agent` instance bound to AgentDeps + AgentMealPlan output.

    `rag_enabled` only shapes the prompt (whether the citation rule is stated);
    the nutrition agent never owns the retriever as a tool.
    """
    prompt = nutrition_agent_system(
        totaller_enabled=totaller_enabled,
        rag_enabled=rag_enabled,
        has_user_request=has_user_request,
    )
    return _build_agent(prompt, totaller_enabled=totaller_enabled, model=model)


def build_refiner_agent(
    *,
    totaller_enabled: bool = True,
    model: Model | None = None,
) -> Agent[AgentDeps, AgentMealPlan]:
    """Refinement agent for the Reflection Loop.

    Same toolset as the main nutrition agent (including the lookup tools, so
    it can swap out an ingredient rather than inventing a replacement), but a
    prompt focused on fixing exactly the issues the critic agent reported
    (see `dietary_advisor.reflection`), not a blind self-review pass.
    """
    return _build_agent(REFLECTION_REFINER_AGENT_SYSTEM, totaller_enabled=totaller_enabled, model=model)
