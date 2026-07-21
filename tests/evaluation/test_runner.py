"""Tests for ablation runner scoring (no live nutrition agent)."""

from __future__ import annotations

import pytest

from dietary_advisor.food_db import FoodDb
from dietary_advisor.planning.hydration import to_food_item
from dietary_advisor.planning.meal_plan import Meal, MealPlan, Portion
from dietary_advisor.planning.pipeline import PipelineResult, VariantConfig
from evaluation.profiles.cases import get_case
from evaluation.runner import run_ablation_grid
from tests.conftest import LONG_INSTRUCTIONS
from tests.evaluation.conftest import agent_plan_single


@pytest.mark.asyncio()
async def test_run_ablation_grid_with_mock_run_fn(food_db: FoodDb, any_code: str) -> None:
    eval_profile = get_case("L1_01")
    targets = eval_profile.profile.targets
    food = to_food_item(food_db.get_food(any_code))

    meal_plan = MealPlan(
        user_id="L1_01",
        meals=[
            Meal(
                kind="lunch",
                name="Real food",
                portions=[Portion(food=food, grams=200.0)],
                recipe=LONG_INSTRUCTIONS,
            ),
        ],
    )
    agent_plan = agent_plan_single(any_code, grams=200.0, user_id="L1_01", food_name=food.name)

    async def fake_run(profile: object, query: str) -> PipelineResult:  # noqa: ARG001
        return PipelineResult(
            plan=meal_plan,
            agent_plan=agent_plan,
            targets=targets,
            iterations=1,
            variant="baseline",
        )

    baseline = VariantConfig(totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    df = await run_ablation_grid(
        variants=[baseline],
        repeats=1,
        run_judge=False,
        run_fn=fake_run,
        lookup=food_db,
    )
    l1 = df[df["case_id"] == "L1_01"]
    assert len(l1) == 1
    row = l1.iloc[0]
    assert row["CSR"] == 1.0
    assert row["error"] is None
    assert "MAE_pct" in df.columns
