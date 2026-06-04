"""Reflection Loop: Generate-Score-Refine.

Implements the kwerenda's "Pętla Walidacyjna": after the LLM produces a
`MealPlan`, deterministically score it; if hard constraints fail, ask the
LLM to refine *with explicit violation feedback* and try again, up to
`reflection_max_loops` iterations. The loop is bounded to keep cost
predictable during the ablation study.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_ai.models import Model

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition_agent import build_refiner_agent, format_violations_prompt
from dietary_advisor.config import get_settings
from dietary_advisor.schemas.constraints import ValidationReport
from dietary_advisor.schemas.meal_plan import MealPlan
from dietary_advisor.validation.validator import validate_meal_plan

log = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Outcome of a Generate-Score-Refine loop."""

    plan: MealPlan
    report: ValidationReport
    iterations: int
    history: list[ValidationReport]


async def reflect_and_refine(
    initial_plan: MealPlan,
    deps: AgentDeps,
    *,
    max_loops: int | None = None,
    model: str | Model | None = None,
) -> ReflectionResult:
    """Iteratively call the refiner agent until validation passes or budget is hit."""
    settings = get_settings()
    budget = max_loops if max_loops is not None else settings.reflection_max_loops

    plan = initial_plan
    report = validate_meal_plan(plan, deps.constraints)
    history: list[ValidationReport] = [report]
    if report.hard_satisfied or budget == 0:
        return ReflectionResult(plan=plan, report=report, iterations=0, history=history)

    refiner = build_refiner_agent(model=model)
    best_plan = plan
    best_report = report
    for i in range(1, budget + 1):
        prompt = format_violations_prompt(best_plan, best_report)
        try:
            result = await refiner.run(prompt, deps=deps)
        except Exception as exc:  # noqa: BLE001 - LLMs raise many things
            log.warning("Refiner agent failed at iteration %d: %s", i, exc)
            break
        candidate = result.output
        candidate_report = validate_meal_plan(candidate, deps.constraints)
        history.append(candidate_report)
        # Accept any iteration that strictly reduces violation count.
        if len(candidate_report.violations) < len(best_report.violations):
            best_plan = candidate
            best_report = candidate_report
        if candidate_report.hard_satisfied:
            return ReflectionResult(
                plan=candidate,
                report=candidate_report,
                iterations=i,
                history=history,
            )

    return ReflectionResult(plan=best_plan, report=best_report, iterations=budget, history=history)
