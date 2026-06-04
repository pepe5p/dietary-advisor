"""Tests for structural integrity and CSR."""

from __future__ import annotations

from dataclasses import replace

from dietary_advisor.schemas.meal_plan import MealKind
from evaluation.profiles.cases import get_case
from evaluation.validation.structural import check_integrity, structural_csr
from tests.evaluation.conftest import (
    agent_plan_rice_lunch,
    agent_plan_with_dinner,
    FDC_PEANUT,
)


def test_check_integrity_empty_meals() -> None:
    plan = agent_plan_rice_lunch()
    plan = plan.model_copy(update={"meals": []})
    assert check_integrity(plan)  # non-empty error list


def test_structural_csr_perfect_vegetarian_plan(mock_lookup: object) -> None:
    plan = agent_plan_rice_lunch(user_id="L2_01")
    eval_profile = get_case("L2_01")
    assert structural_csr(plan, eval_profile, mock_lookup) == 1.0  # type: ignore[arg-type]


def test_structural_csr_fails_on_peanut_allergen(mock_lookup: object) -> None:
    from dietary_advisor.schemas.agent_output import AgentMeal, AgentRecipe, PortionRef

    plan = agent_plan_rice_lunch(user_id="L2_01")
    plan = plan.model_copy(
        update={
            "meals": [
                AgentMeal(
                    kind=MealKind.LUNCH,
                    recipe=AgentRecipe(
                        name="PB rice",
                        portions=[
                            PortionRef(fdc_id=FDC_PEANUT, grams=50.0),
                        ],
                    ),
                ),
            ],
        },
    )
    eval_profile = get_case("L2_01")
    csr = structural_csr(plan, eval_profile, mock_lookup)  # type: ignore[arg-type]
    assert csr < 1.0


def test_structural_csr_forbidden_dinner(mock_lookup: object) -> None:
    plan = agent_plan_with_dinner()
    eval_profile = replace(get_case("L1_01"), forbidden_meal_kinds=frozenset({MealKind.DINNER}))
    assert structural_csr(plan, eval_profile, mock_lookup) == 0.0  # type: ignore[arg-type]
