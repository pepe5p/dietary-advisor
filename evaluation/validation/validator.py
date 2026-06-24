"""Code-based validator: runs the executable rules and emits a `ValidationReport`."""

from __future__ import annotations

from dietary_advisor.schemas.meal_plan import MealPlan
from dietary_advisor.tools.totaller import total_meal_plan
from evaluation.constraints import HardConstraint, ValidationReport
from evaluation.validation.rules import rules_from_constraints


def validate_meal_plan(plan: MealPlan, constraints: list[HardConstraint]) -> ValidationReport:
    """Run every constraint and return a structured report.

    The HSR field is computed as fraction of constraints with zero violations.
    """
    totals = total_meal_plan(plan)
    rules = rules_from_constraints(constraints)
    violations = []
    constraint_passed = 0
    for rule in rules:
        v = rule.check(plan, totals)
        if v:
            violations.extend(v)
        else:
            constraint_passed += 1

    hsr = (constraint_passed / len(rules)) if rules else 1.0
    return ValidationReport(
        hard_satisfied=not violations,
        violations=violations,
        totals=totals.totals,
        hsr=round(hsr, 4),
    )
