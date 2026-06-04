"""Tests for ablation runner scoring (no live nutrition agent)."""

from __future__ import annotations

import pytest

from dietary_advisor.pipeline import PipelineResult, VARIANTS
from dietary_advisor.schemas.constraints import ValidationReport
from dietary_advisor.schemas.meal_plan import Meal, MealKind, MealPlan, Portion, Recipe
from dietary_advisor.tools.tdee import derive_macro_targets
from evaluation.profiles.cases import get_case
from evaluation.runner import run_ablation_grid
from tests.evaluation.conftest import FDC_RICE


@pytest.mark.asyncio()
async def test_run_ablation_grid_with_mock_run_fn(
    rice_food: object,
    mock_lookup: object,
) -> None:
    from dietary_advisor.schemas.nutrition import FoodItem

    food = rice_food  # type: ignore[assignment]
    assert isinstance(food, FoodItem)
    eval_profile = get_case("L1_01")
    targets = derive_macro_targets(eval_profile.profile)

    meal_plan = MealPlan(
        user_id="L1_01",
        meals=[
            Meal(
                kind=MealKind.LUNCH,
                recipe=Recipe(
                    name="Rice",
                    portions=[
                        Portion(food=food.model_copy(update={"fdc_id": FDC_RICE}), grams=200.0),
                    ],
                ),
            ),
        ],
    )

    async def fake_run(profile: object, query: str) -> PipelineResult:  # noqa: ARG001
        return PipelineResult(
            plan=meal_plan,
            report=ValidationReport(hard_satisfied=True, violations=[]),
            targets=targets,
            constraints=list(eval_profile.hard_constraints),
            iterations=1,
            variant="V0",
        )

    df = await run_ablation_grid(
        variants=[VARIANTS["V0"]],
        levels=[1],
        repeats=1,
        run_judge=False,
        run_fn=fake_run,
        lookup=mock_lookup,  # type: ignore[arg-type]
    )
    l1 = df[df["case_id"] == "L1_01"]
    assert len(l1) == 1
    row = l1.iloc[0]
    assert row["CSR"] == 1.0
    assert row["error"] is None
    assert "MAE_pct" in df.columns
