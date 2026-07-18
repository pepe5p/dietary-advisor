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
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.prompts import nutrition_agent_system, reflection_refiner_system
from dietary_advisor.config import get_settings
from dietary_advisor.food_db.facade import LookupQuery
from dietary_advisor.hydration import total_agent_meal_plan
from dietary_advisor.schemas.agent_output import AgentMealPlan

log = logging.getLogger(__name__)


class FoodQuery(BaseModel):
    """One ingredient search in a batch, with independent per-source result caps.

    `max_results_usda`/`max_results_off` let the agent weight a query towards
    the source that actually has it (generic staples in USDA, branded
    products in OFF) instead of searching both identically; 0 skips a source.
    """

    query: str
    max_results_usda: int = Field(default=2, ge=0, le=10)
    max_results_off: int = Field(default=2, ge=0, le=10)


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
            except KeyError:
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
) -> dict[str, list[dict[str, Any]]]:
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
    return await ctx.deps.food_db.lookup([LookupQuery(query, max_results_off, max_results_usda)])


async def lookup_foods(
    ctx: RunContext[AgentDeps],
    queries: list[FoodQuery],
) -> dict[str, list[dict[str, Any]]]:
    """Batch search of both food databases: one tool call for several ingredient queries.

    Each `FoodQuery` sets its own `max_results_usda`/`max_results_off`, so you
    can weight some ingredients towards USDA, others towards OFF, or skip a
    source entirely. Returns hits grouped by source (`open_food_facts`,
    `usda`) exactly like `lookup_food`, pooling matches across every query.
    """
    log.info("lookup_foods(%d query/queries)", len(queries))
    if not queries:
        return {"open_food_facts": [], "usda": []}
    return await ctx.deps.food_db.lookup([LookupQuery(q.query, q.max_results_off, q.max_results_usda) for q in queries])


async def total_meal_plan(ctx: RunContext[AgentDeps], plan: AgentMealPlan) -> dict[str, float]:
    """Deterministically hydrate `plan`'s codes and sum the per-nutrient totals."""
    try:
        totals = total_agent_meal_plan(plan, ctx.deps.food_db)
    except KeyError as exc:
        raise ModelRetry(
            f"Cannot total the plan: {exc}. Every `code` must be copied verbatim from a lookup result.",
        ) from exc
    return {n.value: v for n, v in totals.totals.items()}


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
    """Self-review agent for the Reflection Loop.

    Same toolset as the main nutrition agent (including the lookup tools, so
    it can swap out an ingredient rather than inventing a replacement), but a
    prompt focused on critiquing and improving the previous plan (no
    deterministic feedback - the loop is a plain review/correction pass).
    """
    prompt = reflection_refiner_system(totaller_enabled=totaller_enabled)
    return _build_agent(prompt, totaller_enabled=totaller_enabled)


def format_self_review_prompt(plan: AgentMealPlan, *, totaller_enabled: bool = True) -> str:
    """Render a plain self-review prompt for the refiner agent (no deterministic feedback)."""
    verify = "verify the macro totals with `total_meal_plan`, and " if totaller_enabled else ""
    return (
        "Review the following AgentMealPlan critically: check it against the user's profile "
        f"(allergens, diet pattern, disliked foods), {verify}fix any issues you find.\n\n"
        "Previous AgentMealPlan (JSON):\n"
        f"{plan.model_dump_json(indent=2)}\n\n"
        "Produce an improved AgentMealPlan."
    )
