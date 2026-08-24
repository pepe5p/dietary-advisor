"""Stage 1: deterministic macro error vs frozen targets (Totaller-backed)."""

from dataclasses import dataclass

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.food_db import FoodDb
from dietary_advisor.planning.hydration import total_agent_meal_plan
from dietary_advisor.totaller.nutrition import MacroTargets, NutrientName

_REPORTED_NUTRIENTS = (
    NutrientName.ENERGY_KCAL,
    NutrientName.PROTEIN_G,
    NutrientName.CARBS_G,
    NutrientName.FAT_G,
    NutrientName.FIBER_G,
)
# Fibre is reported per-nutrient but kept out of MAE/MSE: it is the field most
# often absent from a crowd-sourced product record, so its error measures
# database coverage rather than planning accuracy.
_ERROR_NUTRIENTS = frozenset(
    {
        NutrientName.ENERGY_KCAL,
        NutrientName.PROTEIN_G,
        NutrientName.CARBS_G,
        NutrientName.FAT_G,
    },
)


@dataclass(frozen=True)
class NutrientErrors:
    mae: float
    mse: float
    per_nutrient: dict[str, float]


def macro_errors(
    plan: AgentMealPlan,
    targets: MacroTargets,
    lookup: FoodDb,
) -> NutrientErrors:
    """Compute MAE/MSE between Totaller totals and target macros, as % error per nutrient."""
    totals = total_agent_meal_plan(plan, lookup).totals
    target_d = targets.as_dict()
    per: dict[str, float] = {}
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    for nutrient in _REPORTED_NUTRIENTS:
        target = target_d.get(nutrient)
        if target is None or target == 0:
            continue
        actual = float(totals.get(nutrient, 0.0))
        rel = (actual - target) / target * 100.0
        per[nutrient.value] = round(rel, 2)
        if nutrient in _ERROR_NUTRIENTS:
            abs_errors.append(abs(rel))
            sq_errors.append(rel * rel)

    mae = round(sum(abs_errors) / len(abs_errors), 2) if abs_errors else 0.0
    mse = round(sum(sq_errors) / len(sq_errors), 2) if sq_errors else 0.0
    return NutrientErrors(mae=mae, mse=mse, per_nutrient=per)
