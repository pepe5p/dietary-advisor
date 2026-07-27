"""Tests for shared prompt block formatters."""

from __future__ import annotations

from dietary_advisor.agents.prompt_blocks import format_totals_feedback
from dietary_advisor.planning.meal_plan import NutrientTotals
from dietary_advisor.totaller.nutrition import MacroTargets, NutrientName


def test_format_totals_feedback_omits_caveats_when_no_warnings() -> None:
    totals = NutrientTotals(totals={NutrientName.ENERGY_KCAL: 2000.0})
    targets = MacroTargets(energy_kcal=2000.0, protein_g=100.0, carbs_g=250.0, fat_g=60.0)
    text = format_totals_feedback(totals, targets)
    assert "Data-coverage caveats" not in text


def test_format_totals_feedback_includes_caveats() -> None:
    warning = (
        "fiber: 1 of 2 foods have no fiber data - 150 g of 350 g (43% of plan mass), "
        "so the 0.8 g total is understated (Chicken breast)"
    )
    totals = NutrientTotals(
        totals={NutrientName.ENERGY_KCAL: 2000.0, NutrientName.FIBER_G: 0.8},
        warnings=[warning],
    )
    targets = MacroTargets(energy_kcal=2000.0, protein_g=100.0, carbs_g=250.0, fat_g=60.0)
    text = format_totals_feedback(totals, targets)
    assert "Data-coverage caveats (totals above are understated for these nutrients):" in text
    assert f"- {warning}" in text
