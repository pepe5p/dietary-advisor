"""Critic agent prompts: system prompt and user-turn composer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dietary_advisor.agents.prompt_blocks import (
    format_guideline_excerpts,
    join_sections,
    request_sections,
)

if TYPE_CHECKING:
    from dietary_advisor.agents.agent_output import AgentMealPlan
    from dietary_advisor.agents.deps import AgentDeps
    from dietary_advisor.planning.meal_plan import Citation


_CRITIC_INTRO = """You are a meal-plan reviewer. You receive a proposed one-day `AgentMealPlan`,
the user's profile, macro targets{request_clause}. Your only job is to
find genuine problems - you never rewrite the plan yourself.

Check the plan against, in order of severity:"""

_CHECK_ALLERGENS = """Allergens: any ingredient that conflicts with an allergen declared in the
   profile."""

_CHECK_DIET = """Diet pattern: any ingredient violating the profile's diet pattern
   (e.g. meat in a vegan plan)."""

_CHECK_PREFERENCES = "Disliked and preferred foods from the profile."

_CHECK_USER_REQUEST = """The user's original request: does the plan actually deliver what was asked?"""

_CHECK_MACROS = "Macro totals vs targets."

_CHECK_GUARDRAILS_WITH_REQUEST = """Baseline daily guardrails (WHO / DGA 2025-2030), unless a profile condition
   or a cited clinical-guideline excerpt overrides them: fat < 30% of energy,
   saturated fat < 10%, trans fat < 1%; free sugars < 10% of energy and added
   sugars <= 10 g per meal; sodium < 2000 mg/day (hard ceiling 2300 mg);
   potassium >= 3.5 g/day; >= 400 g fruit + vegetables/day. Do NOT flag a
   deviation the user explicitly requested (e.g. crisps with lunch) - instead
   check the rest of the day compensates towards the daily targets."""

_CHECK_GUARDRAILS_NO_REQUEST = """Baseline daily guardrails (WHO / DGA 2025-2030), unless a profile condition
   or a cited clinical-guideline excerpt overrides them: fat < 30% of energy,
   saturated fat < 10%, trans fat < 1%; free sugars < 10% of energy and added
   sugars <= 10 g per meal; sodium < 2000 mg/day (hard ceiling 2300 mg);
   potassium >= 3.5 g/day; >= 400 g fruit + vegetables/day."""

_CHECK_RECIPES = """Recipes: every meal needs full, followable step-by-step instructions that
   match its actual portions."""

_CHECK_RATIONALE = """Rationale: `rationale` must be present and consistent with the actual plan
   (no claims about foods that are not in it), and it must flag supplementation
   where the profile makes a nutrient implausible to cover from food alone."""

_CRITIC_OUTRO = """Report each problem as one short, concrete, actionable issue naming the meal
and ingredient involved (e.g. "Lunch uses feta cheese but the profile is
vegan"). Do NOT nitpick: minor wording, style, or plausible-but-debatable
choices are not issues. If the plan is acceptable, return an empty `issues`
list - that is the expected outcome for a good plan.
"""

_CRITIC_TASK = "Review the plan and return a `PlanCritique`."


def critic_agent_system(*, has_user_request: bool = True) -> str:
    request_clause = ", and original request" if has_user_request else ""
    checks = [
        _CHECK_ALLERGENS,
        _CHECK_DIET,
        _CHECK_PREFERENCES,
    ]
    if has_user_request:
        checks.append(_CHECK_USER_REQUEST)
    checks.extend(
        [
            _CHECK_MACROS,
            _CHECK_GUARDRAILS_WITH_REQUEST if has_user_request else _CHECK_GUARDRAILS_NO_REQUEST,
            _CHECK_RECIPES,
            _CHECK_RATIONALE,
        ],
    )
    numbered = "\n".join(f"{i}. {check}" for i, check in enumerate(checks, start=1))
    return f"{_CRITIC_INTRO.format(request_clause=request_clause)}\n{numbered}\n\n{_CRITIC_OUTRO}"


# Backward-compatible alias for imports that expect a module-level constant.
CRITIC_AGENT_SYSTEM = critic_agent_system()


def critic_user_prompt(
    plan: AgentMealPlan,
    deps: AgentDeps,
    user_query: str,
    *,
    totals_feedback: str | None,
    rag_citations: list[Citation],
) -> str:
    sections = request_sections(deps, user_query)
    if totals_feedback:
        sections.append(totals_feedback)
    excerpts = format_guideline_excerpts(
        rag_citations,
        header="Clinical-guideline excerpts (check the plan's rationale is grounded in these):",
    )
    if excerpts:
        sections.append(excerpts)
    sections.append("Proposed AgentMealPlan (JSON):")
    sections.append(plan.model_dump_json(indent=2))
    sections.append(_CRITIC_TASK)
    return join_sections(sections)
