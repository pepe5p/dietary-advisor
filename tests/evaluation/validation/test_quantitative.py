"""Tests for quantitative (MAE/MSE) evaluation."""

from __future__ import annotations

from dietary_advisor.schemas.nutrition import MacroTargets, NutrientName
from dietary_advisor.tools.food_db import OffFoodDb
from dietary_advisor.tools.totaller import total_agent_meal_plan
from evaluation.validation.quantitative import macro_errors
from tests.evaluation.conftest import agent_plan_single


def test_macro_errors_zero_when_totals_match_targets(off_db: OffFoodDb, any_code: str) -> None:
    plan = agent_plan_single(any_code, grams=200.0)
    totals = total_agent_meal_plan(plan, off_db)
    kcal = totals.totals[NutrientName.ENERGY_KCAL]
    targets = MacroTargets(
        energy_kcal=kcal,
        protein_g=totals.totals.get(NutrientName.PROTEIN_G, 0.0),
        carbs_g=totals.totals.get(NutrientName.CARBS_G, 0.0),
        fat_g=totals.totals.get(NutrientName.FAT_G, 0.0),
        fiber_g=totals.totals.get(NutrientName.FIBER_G, 0.0),
    )
    err = macro_errors(plan, targets, off_db)
    assert err.mae == 0.0
    assert err.mse == 0.0


def test_macro_errors_positive_when_off_target(off_db: OffFoodDb, any_code: str) -> None:
    plan = agent_plan_single(any_code, grams=400.0)
    targets = MacroTargets(energy_kcal=500.0, protein_g=10.0, carbs_g=50.0, fat_g=5.0)
    err = macro_errors(plan, targets, off_db)
    assert err.mae > 0
    assert "energy_kcal" in err.per_nutrient
