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

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition.prompts import nutrition_agent_system, REFLECTION_REFINER_AGENT_SYSTEM
from dietary_advisor.config import get_settings
from dietary_advisor.food_db.errors import MultipleUnknownFoodCodesError
from dietary_advisor.food_db.facade import BatchLookupResult, LookupQuery
from dietary_advisor.planning.hydration import hydrate_meal_plan, total_agent_meal_plan
from dietary_advisor.planning.meal_plan import NutrientTotals

log = logging.getLogger(__name__)


async def _validate_codes(ctx: RunContext[AgentDeps], output: AgentMealPlan) -> AgentMealPlan:
    """Reject any portion whose `code` doesn't resolve against the real food DB.

    The mechanical backstop for "the agent cannot invent a food": a code that
    isn't a genuine lookup result is caught here and sent back as a retry,
    rather than silently reaching hydration.
    """
    try:
        hydrate_meal_plan(output, ctx.deps.food_db)
    except MultipleUnknownFoodCodesError as exc:
        log.error("Unknown food codes: %s", exc.codes)
        raise ModelRetry(
            f"These codes do not exist in the food database: {exc.codes}. "
            "Every `code` must be copied verbatim from a `lookup_foods` result - "
            "search again and use a code you actually received."
        ) from exc
    return output


async def lookup_foods(
    ctx: RunContext[AgentDeps],
    queries: list[LookupQuery],
) -> str:
    """Search both food databases for several ingredient queries in one call.

    Each query runs against both sources. `open_food_facts` holds branded,
    packaged products sold in Poland and stores their names in Polish; `usda`
    holds generic whole foods and reference ingredients (raw carrot, plain
    chicken breast, olive oil) and contains no Polish text. Up to 50 queries
    per call; any beyond that are dropped.

    Returns JSON with one entry per input query, in input order: `query`, then
    `results.open_food_facts` and `results.usda`. Every hit carries `code`,
    `name`, per-100g nutrients in canonical units (`energy_kcal`, `protein_g`,
    `sodium_mg`, ...; unknown ones omitted), plus `brands`, `categories` and
    `ingredients_text` on Open Food Facts hits and `category` on USDA hits.
    """
    log.info("lookup_foods(%d query/queries)", len(queries))
    if not queries:
        return BatchLookupResult().render_json()
    result = await ctx.deps.food_db.lookup(queries)
    return result.render_json()


async def total_meal_plan(ctx: RunContext[AgentDeps], plan: AgentMealPlan) -> NutrientTotals:
    """Sum a draft plan's nutrients from the database rows behind its codes.

    Returns day `totals`, a `per_meal` breakdown in plan order, and `warnings`
    naming nutrients whose totals are understated because some foods carry no
    value for them in the source DB. The plan itself is not modified.
    """
    try:
        return total_agent_meal_plan(plan, ctx.deps.food_db)
    except MultipleUnknownFoodCodesError as exc:
        log.error("Unknown food codes: %s", exc.codes)
        raise ModelRetry(
            f"Cannot total the plan: unknown food code(s) {exc.codes}. "
            "Every `code` must be copied verbatim from a lookup result.",
        ) from exc


def _build_agent(
    system_prompt: str,
    *,
    totaller_enabled: bool,
    model: Model | None = None,
    model_settings: ModelSettings | None = None,
) -> Agent[AgentDeps, AgentMealPlan]:
    settings = get_settings()
    agent: Agent[AgentDeps, AgentMealPlan] = Agent(
        model if model is not None else settings.resolved_llm_model,
        deps_type=AgentDeps,
        output_type=AgentMealPlan,
        system_prompt=system_prompt,
        model_settings=model_settings if model_settings is not None else settings.llm_spec.model_settings,
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
    model_settings: ModelSettings | None = None,
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
    return _build_agent(prompt, totaller_enabled=totaller_enabled, model=model, model_settings=model_settings)


def build_refiner_agent(
    *,
    totaller_enabled: bool = True,
    model: Model | None = None,
    model_settings: ModelSettings | None = None,
) -> Agent[AgentDeps, AgentMealPlan]:
    """Refinement agent for the Reflection Loop.

    Same toolset as the main nutrition agent (including the lookup tools, so
    it can swap out an ingredient rather than inventing a replacement), but a
    prompt focused on fixing exactly the issues the critic agent reported
    (see `dietary_advisor.reflection`), not a blind self-review pass.
    """
    return _build_agent(
        REFLECTION_REFINER_AGENT_SYSTEM,
        totaller_enabled=totaller_enabled,
        model=model,
        model_settings=model_settings,
    )
