"""Meal-idea agent: brainstorms dish concepts ahead of the nutrition agent.

Runs before `nutrition_agent` so ingredient variety follows from a concrete,
varied dish concept chosen up front, rather than from an instruction asking
the nutrition agent to "be varied" while it is already mid-search for a
generic macro-filling role (which tends to collapse onto the same familiar
answer every time). It owns no tools - it never touches the food DB - and its
output is not DB-verified; `nutrition_agent` is free to adapt or discard a
concept if no matching real product exists.
"""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models import Model

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.meal_idea.contract import MealConcept
from dietary_advisor.agents.meal_idea.prompts import MEAL_IDEA_AGENT_SYSTEM
from dietary_advisor.config import get_settings


def build_meal_idea_agent(*, model: Model | None = None) -> Agent[AgentDeps, list[MealConcept]]:
    settings = get_settings()
    return Agent(
        model if model is not None else settings.resolved_llm_model,
        deps_type=AgentDeps,
        output_type=list[MealConcept],
        system_prompt=MEAL_IDEA_AGENT_SYSTEM,
        model_settings=ModelSettings(temperature=settings.meal_idea_llm_temperature),
        retries=1,
    )
