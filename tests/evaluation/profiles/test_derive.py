"""Tests for ground-truth hard-constraint derivation from a `UserProfile`."""

from __future__ import annotations

from dietary_advisor.profile import UserProfile
from dietary_advisor.totaller.nutrition import MacroTargets, NutrientName
from evaluation.profiles.derive import derive_hard_constraints

_TARGETS = MacroTargets(energy_kcal=2200.0, protein_g=120.0, carbs_g=250.0, fat_g=65.0)


def test_derives_allergen_constraints() -> None:
    p = UserProfile(
        user_id="x",
        age=27,
        sex="female",
        height_cm=170,
        weight_kg=60,
        allergens=["peanuts", "milk"],
        diet_pattern="vegan",
        disliked_foods=["mushroom"],
        targets=_TARGETS,
    )
    constraints = derive_hard_constraints(p)
    kinds = {(c.kind, c.target) for c in constraints}
    assert ("allergen_exclusion", "peanuts") in kinds
    assert ("allergen_exclusion", "milk") in kinds
    assert ("diet_pattern", "vegan") in kinds
    assert ("ingredient_exclusion", "mushroom") in kinds


def test_derives_clinical_rules_for_hypertension() -> None:
    p = UserProfile(
        user_id="hyp",
        age=58,
        sex="male",
        height_cm=174,
        weight_kg=95,
        conditions=["hypertension"],
        targets=_TARGETS,
    )
    constraints = derive_hard_constraints(p)
    sodium_rules = [c for c in constraints if c.kind == "max_nutrient" and c.target == NutrientName.SODIUM_MG.value]
    assert sodium_rules
    assert sodium_rules[0].value == 2000.0
