"""Stage 2: response integrity and hard-constraint satisfaction (CSR)."""

from __future__ import annotations

from dietary_advisor.schemas.agent_output import AgentMealPlan
from dietary_advisor.tools.food_db import OffFoodDb
from dietary_advisor.tools.plan_hydrate import hydrate_meal_plan
from evaluation.profiles.eval_profile import EvalProfile
from evaluation.validation.validator import validate_meal_plan


def check_integrity(plan: AgentMealPlan) -> list[str]:
    """Return human-readable integrity violations (empty list => OK)."""
    errors: list[str] = []
    if not plan.meals:
        errors.append("plan has no meals")
        return errors
    for meal in plan.meals:
        if not meal.recipe.portions:
            errors.append(f"{meal.kind.value}: recipe {meal.recipe.name!r} has no portions")
        for ref in meal.recipe.portions:
            if ref.grams <= 0:
                errors.append(f"{meal.kind.value}: non-positive grams for code={ref.code}")
        if not meal.recipe.instructions.strip():
            errors.append(f"{meal.kind.value}: recipe {meal.recipe.name!r} has blank instructions")
    return errors


def structural_csr(plan: AgentMealPlan, eval_profile: EvalProfile, lookup: OffFoodDb) -> float:
    """Constraint Satisfaction Rate from integrity + hard rules + session meal-kind bans."""
    integrity = check_integrity(plan)
    if integrity:
        return 0.0

    for kind in eval_profile.forbidden_meal_kinds:
        if any(m.kind == kind for m in plan.meals):
            return 0.0

    try:
        hydrated = hydrate_meal_plan(plan, lookup)
    except Exception:  # noqa: BLE001
        return 0.0

    report = validate_meal_plan(hydrated, list(eval_profile.hard_constraints))
    if not eval_profile.hard_constraints:
        return 1.0 if report.hard_satisfied else 0.0
    violated = {(v.constraint.kind, v.constraint.target) for v in report.violations}
    passed = sum(1 for c in eval_profile.hard_constraints if (c.kind, c.target) not in violated)
    return round(passed / len(eval_profile.hard_constraints), 4)
