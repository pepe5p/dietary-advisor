"""End-to-end pipeline smoke test using pydantic-ai's `TestModel`.

The build_*_agent factories accept either a string identifier or a concrete
`Model` instance, so we patch them to inject a `TestModel`. This makes the
run fully offline (no LLM API key required) yet still exercises the full
structured-output pipeline (profile is never validated against constraints -
that is an evaluation-only concept, see `evaluation.validation`).
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner

from dietary_advisor.agents import nutrition_agent as na
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.cli import app
from dietary_advisor.config import get_settings
from dietary_advisor.pipeline import Pipeline, VariantConfig
from dietary_advisor.profiles import PROFILES
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

    def patched_nutrition(
        model: str | Model | None = None,  # noqa: ARG001
        *,
        totaller_enabled: bool = True,
    ) -> Agent[AgentDeps, MealPlan]:
        return real_build(model=TestModel(), totaller_enabled=totaller_enabled)

    def patched_refiner(
        model: str | Model | None = None,  # noqa: ARG001
        *,
        totaller_enabled: bool = True,
    ) -> Agent[AgentDeps, MealPlan]:
        return real_build_refiner(model=TestModel(), totaller_enabled=totaller_enabled)

    monkeypatch.setattr(na, "build_nutrition_agent", patched_nutrition)
    monkeypatch.setattr(na, "build_refiner_agent", patched_refiner)
    from dietary_advisor import pipeline as pl

    monkeypatch.setattr(pl, "build_nutrition_agent", patched_nutrition)
    from dietary_advisor.validation import reflection as rl

    monkeypatch.setattr(rl, "build_refiner_agent", patched_refiner)


@pytest.mark.asyncio()
async def test_pipeline_baseline_produces_valid_meal_plan(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _patch_agents_to_test_model(monkeypatch)

    baseline = VariantConfig(food_enabled=False, totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    with Pipeline(baseline) as pipeline:
        result = await pipeline.run(healthy_profile, "Plan one balanced day.")
    assert isinstance(result.plan, MealPlan)
    assert result.plan.user_id


@pytest.mark.asyncio()
async def test_pipeline_runs_with_food_db(
    monkeypatch: pytest.MonkeyPatch,
    vegan_peanut_profile: UserProfile,
) -> None:
    # Enables only the food DB, which reads the real .data/off/off_pl.duckdb.
    if not get_settings().off_db.exists():
        pytest.skip("OFF product DB not found; run `just setup` to build it.")
    _patch_agents_to_test_model(monkeypatch)

    variant = VariantConfig(rag_enabled=False, reflection_enabled=False)
    with Pipeline(variant) as pipeline:
        result = await pipeline.run(vegan_peanut_profile, "Plan a vegan day, no nuts.")
    assert isinstance(result.plan, MealPlan)
    assert result.shopping_list.items


@pytest.mark.asyncio()
async def test_pipeline_chat_threads_message_history(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _patch_agents_to_test_model(monkeypatch)

    baseline = VariantConfig(food_enabled=False, totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    with Pipeline(baseline) as pipeline:
        first = await pipeline.run(healthy_profile, "Plan one balanced day.")
        assert first.messages
        assert first.shopping_list.items
        second = await pipeline.run(
            healthy_profile,
            "Make breakfast lower-carb.",
            message_history=first.messages,
        )

    assert isinstance(second.plan, MealPlan)
    # History accumulates across turns, proving conversation memory is threaded.
    assert len(second.messages) > len(first.messages)


def test_cli_chat_session_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_agents_to_test_model(monkeypatch)
    profile_id = next(iter(PROFILES))  # any built-in profile id

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["chat", "--profile-id", profile_id, "--no-off", "--no-totaller", "--no-rag", "--no-reflective-loop"],
        input="make breakfast lower-carb\nquit\n",
    )
    assert result.exit_code == 0, result.stdout
    assert "Shopping list" in result.stdout
