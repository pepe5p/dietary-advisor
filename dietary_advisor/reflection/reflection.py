"""Reflection Loop: critique-then-refine, gated by a tool-less critic agent.

Implements the kwerenda's "Pętla Walidacyjna" as Self-Refine (iterative
refinement with self-feedback) grounded by a deterministic critic (CRITIC,
Tool-Interactive Critiquing): after the nutrition agent produces an
`AgentMealPlan`, the pipeline itself totals it (when the totaller is
enabled) and hands the plan plus those numbers to a critic agent, which
never edits the plan and only reports concrete issues. The refiner agent
runs only when the critic finds something to fix, and the loop stops as
soon as the critic reports none - so a clean first draft costs one cheap
critic call instead of a fixed number of refiner passes. Bounded by
`reflection_max_loops` to keep worst-case cost predictable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.agents.critic import build_critic_agent, PlanCritique
from dietary_advisor.agents.critic.prompts import critic_user_prompt
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition import build_refiner_agent
from dietary_advisor.agents.nutrition.prompts import refiner_user_prompt
from dietary_advisor.agents.prompt_blocks import format_totals_feedback
from dietary_advisor.agents.prompt_blocks import has_user_request as user_has_request
from dietary_advisor.agents.runner import run_agent_logged
from dietary_advisor.config import get_settings
from dietary_advisor.food_db.errors import MultipleUnknownFoodCodesError
from dietary_advisor.planning.hydration import total_agent_meal_plan
from dietary_advisor.planning.meal_plan import Citation
from dietary_advisor.telemetry import collect_from_result, RunTelemetry

log = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Outcome of a critique-then-refine loop."""

    plan: AgentMealPlan
    # Number of completed refiner passes (0 if the critic approved the first draft).
    iterations: int
    # True iff the critic signed off (empty issues); False if the loop ran out of budget.
    approved: bool = False
    telemetry: RunTelemetry = field(default_factory=RunTelemetry)


def _totals_feedback_for(plan: AgentMealPlan, deps: AgentDeps, *, iteration: int) -> str | None:
    """Deterministically total `plan`, or `None` if it can't be hydrated (logged, never fatal)."""
    try:
        totals = total_agent_meal_plan(plan, deps.food_db)
    except MultipleUnknownFoodCodesError as exc:
        log.error(
            "Could not compute deterministic totals at iteration %d; unknown food codes: %s",
            iteration,
            exc.codes,
        )
        return None
    return format_totals_feedback(totals, deps.targets)


async def reflect_and_refine(
    initial_plan: AgentMealPlan,
    deps: AgentDeps,
    user_query: str,
    *,
    max_loops: int | None = None,
    totaller_enabled: bool = True,
    rag_citations: list[Citation] | None = None,
    model: Model | None = None,
    model_settings: ModelSettings | None = None,
    has_user_request: bool | None = None,
) -> ReflectionResult:
    """Run up to `max_loops` critique-then-refine passes, stopping once the critic reports no issues."""
    settings = get_settings()
    budget = max_loops if max_loops is not None else settings.reflection_max_loops
    if budget == 0:
        return ReflectionResult(plan=initial_plan, iterations=0)

    request_present = has_user_request if has_user_request is not None else user_has_request(user_query)
    citations = rag_citations or []
    critic = build_critic_agent(model=model, model_settings=model_settings, has_user_request=request_present)
    refiner = build_refiner_agent(totaller_enabled=totaller_enabled, model=model, model_settings=model_settings)
    plan = initial_plan
    telemetry = RunTelemetry()

    for i in range(1, budget + 1):
        totals_feedback = _totals_feedback_for(plan, deps, iteration=i) if totaller_enabled else None

        critic_prompt = critic_user_prompt(
            plan,
            deps,
            user_query,
            totals_feedback=totals_feedback,
            rag_citations=citations,
        )
        try:
            critic_result = await run_agent_logged(critic, critic_prompt, deps=deps, label=f"critic#{i}")
        except Exception as exc:  # noqa: BLE001 - LLMs raise many things
            log.warning("Critic agent failed at iteration %d: %s", i, exc)
            return ReflectionResult(plan=plan, iterations=i - 1, telemetry=telemetry)
        telemetry = telemetry.merge(collect_from_result(critic_result))
        critique: PlanCritique = critic_result.output

        if not critique.issues:
            log.info("Critic approved the plan after %d refiner iteration(s).", i - 1)
            return ReflectionResult(plan=plan, iterations=i - 1, approved=True, telemetry=telemetry)
        log.info("Critic reported %d issue(s) at iteration %d: %s", len(critique.issues), i, critique.issues)

        refiner_prompt = refiner_user_prompt(
            plan,
            deps,
            user_query,
            issues=critique.issues,
            totals_feedback=totals_feedback,
            rag_citations=citations,
        )
        refiner_result = await run_agent_logged(refiner, refiner_prompt, deps=deps, label=f"refiner#{i}")
        plan = refiner_result.output
        telemetry = telemetry.merge(collect_from_result(refiner_result))

    return ReflectionResult(plan=plan, iterations=budget, telemetry=telemetry)
