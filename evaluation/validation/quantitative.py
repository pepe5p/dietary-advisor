"""Stage 1: deterministic macro error vs frozen targets (Totaller-backed)."""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.schemas.agent_output import AgentMealPlan
from dietary_advisor.schemas.nutrition import MacroTargets, NutrientName
from dietary_advisor.tools.food_db import OffFoodDb
from dietary_advisor.tools.totaller import total_agent_meal_plan

_MACRO_NUTRIENTS = (
    NutrientName.ENERGY_KCAL,
    NutrientName.PROTEIN_G,
    NutrientName.CARBS_G,
    NutrientName.FAT_G,
    NutrientName.FIBER_G,
)


@dataclass(frozen=True)
class NutrientErrors:
    mae: float
    mse: float
    per_nutrient: dict[str, float]


def macro_errors(
    plan: AgentMealPlan,
    targets: MacroTargets,
    lookup: OffFoodDb,
) -> NutrientErrors:
    """Compute MAE/MSE between Totaller totals and target macros, as % error per nutrient."""
    totals = total_agent_meal_plan(plan, lookup).totals
    target_d = targets.as_dict()
    per: dict[str, float] = {}
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    for nutrient in _MACRO_NUTRIENTS:
        target = target_d.get(nutrient)
        if target is None or target == 0:
            continue
        actual = float(totals.get(nutrient, 0.0))
        rel = (actual - target) / target * 100.0
        per[nutrient.value] = round(rel, 2)
        abs_errors.append(abs(rel))
        sq_errors.append(rel * rel)

    mae = round(sum(abs_errors) / len(abs_errors), 2) if abs_errors else 0.0
    mse = round(sum(sq_errors) / len(sq_errors), 2) if sq_errors else 0.0
    return NutrientErrors(mae=mae, mse=mse, per_nutrient=per)
