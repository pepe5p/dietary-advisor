"""Sanity checks on the evaluation-only constraint schema."""

from __future__ import annotations

from dietary_advisor.schemas.nutrition import NutrientName
from evaluation.constraints import HardConstraint


def test_hard_constraint_factories() -> None:
    c = HardConstraint.allergen("peanuts")
    assert c.kind == "allergen_exclusion"
    assert c.target == "peanuts"

    c2 = HardConstraint.max_nutrient(NutrientName.SODIUM_MG, value=2000)
    assert c2.kind == "max_nutrient"
    assert c2.value == 2000
