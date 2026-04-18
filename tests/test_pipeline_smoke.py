"""End-to-end pipeline smoke test using pydantic-ai's `TestModel`.

The build_*_agent factories accept either a string identifier or a concrete
`Model` instance, so we patch them to inject a `TestModel`. This makes the
run fully offline (no LLM API key required) yet still exercises the full
structured-output + deterministic-validator pipeline.
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel

from dietary_advisor.agents import nutrition_agent as na
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.pipeline import Pipeline, VARIANTS
from dietary_advisor.profile_manager.service import ProfileService
from dietary_advisor.profile_manager.store import ProfileStore
from dietary_advisor.schemas.meal_plan import MealPlan
from dietary_advisor.schemas.profile import UserProfile


def _patch_agents_to_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the build_*_agent factories so they always use `TestModel`.

    Patches BOTH `dietary_advisor.agents.nutrition_agent` (the source) and
    `dietary_advisor.pipeline` (the import site) since `pipeline` imported
    the symbol at module load time.
    """
    real_build = na.build_nutrition_agent
    real_build_refiner = na.build_refiner_agent

    def patched_nutrition(model: str | Model | None = None) -> Agent[AgentDeps, MealPlan]:  # noqa: ARG001
        return real_build(model=TestModel())

    def patched_refiner(model: str | Model | None = None) -> Agent[AgentDeps, MealPlan]:  # noqa: ARG001
        return real_build_refiner(model=TestModel())

    monkeypatch.setattr(na, "build_nutrition_agent", patched_nutrition)
    monkeypatch.setattr(na, "build_refiner_agent", patched_refiner)
    from dietary_advisor import pipeline as pl

    monkeypatch.setattr(pl, "build_nutrition_agent", patched_nutrition)
    from dietary_advisor.validation import reflection as rl

    monkeypatch.setattr(rl, "build_refiner_agent", patched_refiner)


@pytest.mark.asyncio()
async def test_pipeline_v0_produces_valid_meal_plan(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _patch_agents_to_test_model(monkeypatch)

    store = ProfileStore()
    store.upsert(healthy_profile)
    service = ProfileService(store=store)

    with Pipeline(VARIANTS["V0"], profile_service=service) as pipeline:
        result = await pipeline.run(healthy_profile, "Plan one balanced day.")
    assert isinstance(result.plan, MealPlan)
    assert result.plan.user_id
    # V0 has no constraints, so the report should be vacuously satisfied.
    assert result.report.hard_satisfied is True


@pytest.mark.asyncio()
async def test_pipeline_v2_runs_with_constraints(
    monkeypatch: pytest.MonkeyPatch,
    vegan_peanut_profile: UserProfile,
) -> None:
    _patch_agents_to_test_model(monkeypatch)

    store = ProfileStore()
    store.upsert(vegan_peanut_profile)
    service = ProfileService(store=store)

    with Pipeline(VARIANTS["V2"], profile_service=service) as pipeline:
        result = await pipeline.run(vegan_peanut_profile, "Plan a vegan day, no nuts.")
    # The TestModel-generated MealPlan does not respect constraints by
    # design; what we verify is that the validator correctly produced a
    # report and the constraint set was derived from the profile.
    assert result.constraints
    assert isinstance(result.plan, MealPlan)
