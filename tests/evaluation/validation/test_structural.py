"""Tests for structural integrity and CSR."""

from __future__ import annotations

from dataclasses import replace

from dietary_advisor.schemas.meal_plan import MealKind
from dietary_advisor.tools.food_db import OffFoodDb
from evaluation.profiles.cases import get_case
from evaluation.validation.structural import check_integrity, structural_csr
from tests.evaluation.conftest import agent_plan_rice_lunch, agent_plan_single, agent_plan_with_dinner


def test_check_integrity_empty_meals() -> None:
    plan = agent_plan_rice_lunch()
    plan = plan.model_copy(update={"meals": []})
    assert check_integrity(plan)  # non-empty error list


def test_structural_csr_perfect_vegetarian_plan(off_db: OffFoodDb, vegetarian_code: str) -> None:
    plan = agent_plan_single(vegetarian_code, user_id="L2_01")
    eval_profile = get_case("L2_01")
    assert structural_csr(plan, eval_profile, off_db) == 1.0


def test_structural_csr_fails_on_peanut_allergen(off_db: OffFoodDb, peanut_code: str) -> None:
    plan = agent_plan_single(peanut_code, user_id="L2_01", name="PB dish", grams=50.0)
    eval_profile = get_case("L2_01")
    assert structural_csr(plan, eval_profile, off_db) < 1.0


def test_structural_csr_forbidden_dinner(off_db: OffFoodDb, any_code: str) -> None:
    plan = agent_plan_with_dinner(any_code)
    eval_profile = replace(get_case("L1_01"), forbidden_meal_kinds=frozenset({MealKind.DINNER}))
    assert structural_csr(plan, eval_profile, off_db) == 0.0
