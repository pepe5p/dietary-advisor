"""Meal-idea agent prompts: system prompt and user-turn composer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dietary_advisor.agents.prompt_blocks import (
    format_guideline_excerpts,
    join_sections,
    request_sections,
)

if TYPE_CHECKING:
    from dietary_advisor.agents.deps import AgentDeps
    from dietary_advisor.planning.meal_plan import Citation


MEAL_IDEA_AGENT_SYSTEM = """You are a meal-idea generator. For each meal slot of the day (breakfast,
lunch, dinner, and a snack or two), suggest THREE concrete dish names as
alternatives for that one slot - three ways to eat that breakfast, not three
breakfasts to eat. Another agent picks one option per slot and turns it into an
actual plan; that is the whole division of labour.

Name specific dishes, not nutrient-role placeholders - "Turkish menemen with
feta", not "a high-protein breakfast". Make a slot's three options genuinely
different from one another rather than three variations on one idea.

You are suggesting dishes for a person living in Poland, so every dish must be
buildable from ingredients and products actually available there. The dishes do
not need to be Polish - everyday dishes from any cuisine are fine. Favour
varied, non-obvious ideas over the first predictable option, but keep that
variety within what is obtainable in Poland: no obscure specialties whose
ingredients the person is unlikely to want or able to find.
"""

_MEAL_IDEA_TASK = "Propose three dish options for each meal slot."


def meal_idea_user_prompt(deps: AgentDeps, user_query: str, *, rag_citations: list[Citation]) -> str:
    sections = request_sections(deps, user_query)
    excerpts = format_guideline_excerpts(
        rag_citations,
        header="Clinical-guideline excerpts (keep the dish concepts consistent with these):",
    )
    if excerpts:
        sections.append(excerpts)
    sections.append(_MEAL_IDEA_TASK)
    return join_sections(sections)
