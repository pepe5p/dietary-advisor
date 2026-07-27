"""Tests for collect_runs skip-existing and failure-retry behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from dietary_advisor.food_db import FoodDb
from dietary_advisor.planning.hydration import to_food_item
from dietary_advisor.planning.meal_plan import Meal, MealPlan, Portion
from dietary_advisor.planning.pipeline import PipelineResult, VariantConfig
from dietary_advisor.profiles import get_profile
from dietary_advisor.totaller.nutrition import MacroTargets
from evaluation.case_runner.collect import collect_runs
from evaluation.case_runner.grid import RunSpec
from evaluation.case_runner.store import is_done, load
from tests.conftest import LONG_INSTRUCTIONS
from tests.evaluation.conftest import agent_plan_single


@pytest.mark.asyncio()
async def test_collect_runs_skips_existing_and_saves_new(tmp_path: Path, food_db: FoodDb, any_code: str) -> None:
    profile = get_profile("regular")
    targets = profile.targets
    food = to_food_item(food_db.get_food(any_code))
    meal_plan = MealPlan(
        user_id="regular",
        meals=[
            Meal(
                kind="lunch",
                name="Real food",
                portions=[Portion(food=food, grams=200.0)],
                recipe=LONG_INSTRUCTIONS,
            ),
        ],
    )
    agent_plan = agent_plan_single(any_code, grams=200.0, user_id="regular", food_name=food.name)
    variant = VariantConfig(totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    specs = [
        RunSpec(llm_model="test-model", variant=variant, scenario_id="regular"),
        RunSpec(llm_model="test-model", variant=variant, scenario_id="preferences"),
    ]

    calls: list[str] = []

    async def fake_run(profile: object, query: str) -> PipelineResult:  # noqa: ARG001
        calls.append(query)
        return PipelineResult(
            plan=meal_plan,
            agent_plan=agent_plan,
            targets=targets,
            iterations=0,
            variant=variant.label,
        )

    # Pre-seed regular so it is skipped.
    first = await collect_runs([specs[0]], output_dir=tmp_path, run_fn=fake_run)
    assert first.succeeded == 1
    assert is_done(specs[0], output_dir=tmp_path)
    assert len(calls) == 1

    summary = await collect_runs(specs, output_dir=tmp_path, run_fn=fake_run)
    assert summary.planned == 2
    assert summary.already_done == 1
    assert summary.attempted == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert len(calls) == 2
    assert is_done(specs[1], output_dir=tmp_path)
    loaded = load(specs[1], output_dir=tmp_path)
    assert loaded.scenario_id == "preferences"
    assert loaded.targets == MacroTargets(
        energy_kcal=targets.energy_kcal,
        protein_g=targets.protein_g,
        carbs_g=targets.carbs_g,
        fat_g=targets.fat_g,
        fiber_g=targets.fiber_g,
    )


@pytest.mark.asyncio()
async def test_collect_runs_failure_leaves_no_file(tmp_path: Path) -> None:
    variant = VariantConfig()
    spec = RunSpec(llm_model="test-model", variant=variant, scenario_id="regular")

    async def boom(profile: object, query: str) -> PipelineResult:  # noqa: ARG001
        raise RuntimeError("llm down")

    summary = await collect_runs([spec], output_dir=tmp_path, run_fn=boom)
    assert summary.failed == 1
    assert summary.succeeded == 0
    assert not is_done(spec, output_dir=tmp_path)
