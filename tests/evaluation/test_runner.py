"""Tests for ablation runner scoring (no live nutrition agent)."""

from __future__ import annotations

import pytest

from dietary_advisor.pipeline import PipelineResult, VariantConfig
from dietary_advisor.schemas.meal_plan import Meal, MealKind, MealPlan, Portion, Recipe
from dietary_advisor.tools.food_db import OffFoodDb
from evaluation.profiles.cases import get_case
from evaluation.runner import run_ablation_grid
from tests.conftest import LONG_INSTRUCTIONS


@pytest.mark.asyncio()
async def test_run_ablation_grid_with_mock_run_fn(off_db: OffFoodDb, any_code: str) -> None:
    eval_profile = get_case("L1_01")
    targets = eval_profile.profile.targets
    food = off_db.get_food(any_code)

    meal_plan = MealPlan(
        user_id="L1_01",
        meals=[
            Meal(
                kind=MealKind.LUNCH,
                recipe=Recipe(
                    name="Real food",
                    portions=[Portion(food=food, grams=200.0)],
                    instructions=LONG_INSTRUCTIONS,
                ),
            ),
        ],
    )

    async def fake_run(profile: object, query: str) -> PipelineResult:  # noqa: ARG001
        return PipelineResult(
            plan=meal_plan,
            targets=targets,
            iterations=1,
            variant="baseline",
        )

    baseline = VariantConfig(food_enabled=False, totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    df = await run_ablation_grid(
        variants=[baseline],
        levels=[1],
        repeats=1,
        run_judge=False,
        run_fn=fake_run,
        lookup=off_db,
    )
    l1 = df[df["case_id"] == "L1_01"]
    assert len(l1) == 1
    row = l1.iloc[0]
    assert row["CSR"] == 1.0
    assert row["error"] is None
    assert "MAE_pct" in df.columns
