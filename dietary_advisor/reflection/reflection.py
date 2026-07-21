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

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.agents.critic_agent import build_critic_agent
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition_agent import build_refiner_agent
from dietary_advisor.agents.prompts import format_guideline_excerpts
from dietary_advisor.agents.reflection import PlanCritique
from dietary_advisor.agents.runner import run_agent_logged
from dietary_advisor.config import get_settings
from dietary_advisor.food_db.errors import UnknownFoodCodeError
from dietary_advisor.planning.hydration import total_agent_meal_plan
from dietary_advisor.planning.meal_plan import Citation, NutrientTotals
from dietary_advisor.telemetry import collect_from_result, RunTelemetry
from dietary_advisor.totaller.nutrition import MacroTargets, NutrientName

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


def _fmt_pct(actual: float, target: float) -> str:
    if target <= 0:
        return "n/a"
    return f"{(actual - target) / target * 100:+.0f}%"


def _format_totals_feedback(totals: NutrientTotals, targets: MacroTargets) -> str:
    """Render the deterministic totals-vs-targets block injected into critic/refiner prompts."""
    t = totals.totals
    target_map = targets.as_dict()

    def part(name: NutrientName, label: str, unit: str) -> str:
        actual = t.get(name, 0.0)
        target = target_map[name]
        return f"{label} {actual:.0f} {unit} (target {target:.0f}, {_fmt_pct(actual, target)})"

    overall = ", ".join(
        [
            part(NutrientName.ENERGY_KCAL, "", "kcal").strip(),
            part(NutrientName.PROTEIN_G, "protein", "g"),
            part(NutrientName.CARBS_G, "carbs", "g"),
            part(NutrientName.FAT_G, "fat", "g"),
            part(NutrientName.FIBER_G, "fiber", "g"),
        ]
    )
    per_meal_lines = [
        (
            f'- {meal.kind} "{meal.name}": {meal.get(NutrientName.ENERGY_KCAL):.0f} kcal, '
            f"protein {meal.get(NutrientName.PROTEIN_G):.0f} g, "
            f"carbs {meal.get(NutrientName.CARBS_G):.0f} g, "
            f"fat {meal.get(NutrientName.FAT_G):.0f} g"
        )
        for meal in totals.per_meal
    ]
    return "\n".join(
        [
            "Deterministic nutrient totals (computed by the system, trust these numbers):",
            f"Overall: {overall}",
            "Per meal:",
            *per_meal_lines,
        ]
    )


def _format_critic_prompt(
    plan: AgentMealPlan,
    deps: AgentDeps,
    user_query: str,
    totals_feedback: str | None,
    rag_citations: list[Citation],
) -> str:
    sections = [
        f"Original user request: {user_query}",
        "Profile:",
        deps.profile.model_dump_json(indent=2),
        "Macro targets (single day):",
        deps.targets.model_dump_json(indent=2),
    ]
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
    sections.append("Review the plan and return a `PlanCritique`.")
    return "\n\n".join(sections)


def _format_refiner_prompt(
    plan: AgentMealPlan,
    deps: AgentDeps,
    user_query: str,
    issues: list[str],
    totals_feedback: str | None,
    rag_citations: list[Citation],
) -> str:
    sections = [
        f"Original user request: {user_query}",
        "Profile:",
        deps.profile.model_dump_json(indent=2),
        "Macro targets (single day):",
        deps.targets.model_dump_json(indent=2),
        "Reviewer issues to fix:",
        "\n".join(f"- {issue}" for issue in issues),
    ]
    if totals_feedback:
        sections.append(totals_feedback)
    excerpts = format_guideline_excerpts(
        rag_citations,
        header="Clinical-guideline excerpts (keep citations accurate to these):",
    )
    if excerpts:
        sections.append(excerpts)
    sections.append("Previous AgentMealPlan (JSON):")
    sections.append(plan.model_dump_json(indent=2))
    sections.append("Fix the listed issues with minimal targeted changes and return the full, updated AgentMealPlan.")
    return "\n\n".join(sections)


def _totals_feedback_for(plan: AgentMealPlan, deps: AgentDeps, *, iteration: int) -> str | None:
    """Deterministically total `plan`, or `None` if it can't be hydrated (logged, never fatal)."""
    try:
        totals = total_agent_meal_plan(plan, deps.food_db)
    except UnknownFoodCodeError as exc:
        log.warning("Could not compute deterministic totals at iteration %d: %s", iteration, exc)
        return None
    return _format_totals_feedback(totals, deps.targets)


async def reflect_and_refine(
    initial_plan: AgentMealPlan,
    deps: AgentDeps,
    user_query: str,
    *,
    max_loops: int | None = None,
    totaller_enabled: bool = True,
    rag_citations: list[Citation] | None = None,
) -> ReflectionResult:
    """Run up to `max_loops` critique-then-refine passes, stopping once the critic reports no issues."""
    settings = get_settings()
    budget = max_loops if max_loops is not None else settings.reflection_max_loops
    if budget == 0:
        return ReflectionResult(plan=initial_plan, iterations=0)

    citations = rag_citations or []
    critic = build_critic_agent()
    refiner = build_refiner_agent(totaller_enabled=totaller_enabled)
    plan = initial_plan
    telemetry = RunTelemetry()

    for i in range(1, budget + 1):
        totals_feedback = _totals_feedback_for(plan, deps, iteration=i) if totaller_enabled else None

        critic_prompt = _format_critic_prompt(plan, deps, user_query, totals_feedback, citations)
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

        refiner_prompt = _format_refiner_prompt(plan, deps, user_query, critique.issues, totals_feedback, citations)
        try:
            refiner_result = await run_agent_logged(refiner, refiner_prompt, deps=deps, label=f"refiner#{i}")
        except Exception as exc:  # noqa: BLE001 - LLMs raise many things
            log.warning("Refiner agent failed at iteration %d: %s", i, exc)
            return ReflectionResult(plan=plan, iterations=i - 1, telemetry=telemetry)
        plan = refiner_result.output
        telemetry = telemetry.merge(collect_from_result(refiner_result))

    return ReflectionResult(plan=plan, iterations=budget, telemetry=telemetry)
