"""Reflection Loop: plain Generate-Review-Refine.

Implements the kwerenda's "Pętla Walidacyjna" as a constraint-free
self-review pass: after the LLM produces a `MealPlan`, ask a (typically
stronger) refiner agent to critique and improve it, up to
`reflection_max_loops` iterations. There is no deterministic scorer here -
hard constraints are an evaluation-only concept the production pipeline never
sees, so this loop cannot check whether an iteration is "better", only that
each pass runs. The loop is bounded to keep cost predictable during the
ablation study.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic_ai.models import Model

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition_agent import build_refiner_agent, format_self_review_prompt
from dietary_advisor.agents.runner import run_agent_logged
from dietary_advisor.config import get_settings
from dietary_advisor.schemas.meal_plan import MealPlan
from dietary_advisor.telemetry import collect_from_result, RunTelemetry

log = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Outcome of a Generate-Review-Refine loop."""

    plan: MealPlan
    iterations: int
    telemetry: RunTelemetry = field(default_factory=RunTelemetry)


async def reflect_and_refine(
    initial_plan: MealPlan,
    deps: AgentDeps,
    *,
    max_loops: int | None = None,
    model: str | Model | None = None,
    totaller_enabled: bool = True,
) -> ReflectionResult:
    """Run up to `max_loops` self-review passes over the plan, keeping the last successful revision."""
    settings = get_settings()
    budget = max_loops if max_loops is not None else settings.reflection_max_loops
    if budget == 0:
        return ReflectionResult(plan=initial_plan, iterations=0)

    refiner = build_refiner_agent(model=model, totaller_enabled=totaller_enabled)
    plan = initial_plan
    telemetry = RunTelemetry()
    for i in range(1, budget + 1):
        prompt = format_self_review_prompt(plan)
        try:
            result = await run_agent_logged(refiner, prompt, deps=deps, label=f"refiner#{i}")
        except Exception as exc:  # noqa: BLE001 - LLMs raise many things
            log.warning("Refiner agent failed at iteration %d: %s", i, exc)
            return ReflectionResult(plan=plan, iterations=i - 1, telemetry=telemetry)
        plan = result.output
        telemetry = telemetry.merge(collect_from_result(result))

    return ReflectionResult(plan=plan, iterations=budget, telemetry=telemetry)
