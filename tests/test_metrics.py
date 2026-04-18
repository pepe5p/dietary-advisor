"""Tests for evaluation/metrics.py."""

from __future__ import annotations

from dietary_advisor.schemas.constraints import HardConstraint, ValidationReport, Violation
from dietary_advisor.schemas.meal_plan import (
    Citation,
    Meal,
    MealKind,
    MealPlan,
    Portion,
    Recipe,
)
from dietary_advisor.schemas.nutrition import FoodItem, MacroTargets, NutrientName
from dietary_advisor.schemas.profile import Sex, UserProfile
from evaluation.metrics import csr, faithfulness, hsr, nutrient_errors, ssr


def _plan(food: FoodItem, grams: float = 100.0) -> MealPlan:
    return MealPlan(
        user_id="x",
        meals=[
            Meal(
                kind=MealKind.LUNCH,
                recipe=Recipe(
                    name="r",
                    portions=[Portion(food=food, grams=grams)],
                ),
            )
        ],
    )


def test_hsr_no_violations() -> None:
    report = ValidationReport(hard_satisfied=True, violations=[])
    assert hsr(report, []) == 1.0
    assert hsr(report, [HardConstraint.allergen("milk")]) == 1.0


def test_hsr_with_violation() -> None:
    c = HardConstraint.allergen("peanuts")
    report = ValidationReport(
        hard_satisfied=False,
        violations=[Violation(constraint=c, detail="d")],
    )
    assert hsr(report, [c, HardConstraint.allergen("milk")]) == 0.5


def test_ssr_preferred_and_disliked(rice_food: FoodItem) -> None:
    profile = UserProfile(
        user_id="x",
        age=30,
        sex=Sex.MALE,
        height_cm=180,
        weight_kg=78,
        preferred_foods=["rice"],
        disliked_foods=["beef"],
    )
    plan = _plan(rice_food, 200)
    assert ssr(plan, profile) == 1.0


def test_ssr_partial_preferred(rice_food: FoodItem) -> None:
    profile = UserProfile(
        user_id="x",
        age=30,
        sex=Sex.MALE,
        height_cm=180,
        weight_kg=78,
        preferred_foods=["rice", "salmon"],
    )
    plan = _plan(rice_food)
    assert ssr(plan, profile) == 0.5


def test_csr_blends_h_and_s(rice_food: FoodItem) -> None:
    profile = UserProfile(
        user_id="x",
        age=30,
        sex=Sex.MALE,
        height_cm=180,
        weight_kg=78,
        preferred_foods=["beef"],
    )
    plan = _plan(rice_food)
    report = ValidationReport(hard_satisfied=True, violations=[])
    val = csr(report, plan, profile, [], hard_weight=0.5)
    assert 0 <= val <= 1


def test_nutrient_errors_match_target() -> None:
    targets = MacroTargets(energy_kcal=2000, protein_g=120, carbs_g=220, fat_g=65, fiber_g=25)
    report = ValidationReport(
        hard_satisfied=True,
        totals={
            NutrientName.ENERGY_KCAL: 2000,
            NutrientName.PROTEIN_G: 120,
            NutrientName.CARBS_G: 220,
            NutrientName.FAT_G: 65,
            NutrientName.FIBER_G: 25,
        },
    )
    err = nutrient_errors(report, targets)
    assert err.mae == 0.0
    assert err.mse == 0.0


def test_nutrient_errors_off_target() -> None:
    targets = MacroTargets(energy_kcal=2000, protein_g=100, carbs_g=200, fat_g=60)
    report = ValidationReport(
        hard_satisfied=True,
        totals={
            NutrientName.ENERGY_KCAL: 2200,
            NutrientName.PROTEIN_G: 80,
            NutrientName.CARBS_G: 200,
            NutrientName.FAT_G: 60,
        },
    )
    err = nutrient_errors(report, targets)
    assert err.mae > 0
    assert "energy_kcal" in err.per_nutrient


def test_faithfulness_no_rationale_is_perfect(rice_food: FoodItem) -> None:
    plan = _plan(rice_food)
    plan.rationale = ""
    assert faithfulness(plan) == 1.0


def test_faithfulness_with_supporting_citation(rice_food: FoodItem) -> None:
    plan = _plan(rice_food)
    plan.rationale = "Whole grains and dietary fibre contribute to glycaemic control."
    plan.citations = [
        Citation(
            source="ADA_NUTRITION_2019",
            snippet="People with diabetes should consume the recommended amount of fibre and whole grains.",
        ),
    ]
    score = faithfulness(plan)
    assert score == 1.0


def test_faithfulness_zero_when_no_citations(rice_food: FoodItem) -> None:
    plan = _plan(rice_food)
    plan.rationale = "This sentence is not supported."
    plan.citations = []
    assert faithfulness(plan) == 0.0
