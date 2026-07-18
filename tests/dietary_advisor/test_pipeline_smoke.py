"""End-to-end pipeline smoke test using pydantic-ai's `TestModel`.

`Settings.resolved_llm_model` is a cached property, so pre-seeding its cache
with a `TestModel` makes every agent factory (which all read from settings)
pick it up. This makes the run fully offline (no LLM API key required) yet
still exercises the full structured-output pipeline (profile is never
validated against constraints - that is an evaluation-only concept, see
`evaluation.validation`).

`TestModel` fabricates a `PortionRef.code` that satisfies the JSON schema but
has no relation to a real product, and the agent's output validator rejects
any code that doesn't resolve - so these tests inject `_FakeFoodDb`, a
permissive stand-in that resolves any code, rather than depending on
`.data/off/off_pl.duckdb` actually being built.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner

import dietary_advisor.pipeline as pipeline_module
from dietary_advisor.cli import app
from dietary_advisor.config import get_settings
from dietary_advisor.food_db import OFFItem
from dietary_advisor.food_db.facade import LookupQuery
from dietary_advisor.pipeline import Pipeline, VariantConfig
from dietary_advisor.profiles import PROFILES
from dietary_advisor.schemas.blueprint import MealConcept
from dietary_advisor.schemas.meal_plan import MealPlan
from dietary_advisor.schemas.nutrition import NutrientName
from dietary_advisor.schemas.profile import UserProfile


class _FakeFoodDb:
    """Permissive stand-in for `FoodDb`: resolves any code, always finds a hit.

    Also exposes an `open` classmethod so it can substitute for the `FoodDb`
    class itself (see `_use_fake_food_db`), matching `FoodDb.open(settings)`.
    """

    @classmethod
    def open(cls, settings: object = None) -> _FakeFoodDb:  # noqa: ARG003
        return cls()

    async def lookup(self, queries: Sequence[LookupQuery]) -> dict[str, list[dict[str, Any]]]:  # noqa: ARG002
        hits: list[dict[str, Any]] = [
            {
                "code": "test:1",
                "name": "Test food",
                "brands": None,
                "categories": None,
                "ingredients_text": None,
                "tags": [],
                "kcal_per_100g": 200.0,
                "protein_g_per_100g": 10.0,
                "carbs_g_per_100g": 20.0,
                "fat_g_per_100g": 5.0,
            },
        ]
        return {"open_food_facts": hits, "usda": []}

    def get_food(self, code: str) -> OFFItem:
        return OFFItem(code=code, name="Test food", nutrients_per_100g={NutrientName.ENERGY_KCAL: 200.0})

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeFoodDb:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _use_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-seed `resolved_llm_model`'s cache so its real (API-backed) resolution never runs."""
    monkeypatch.setitem(get_settings().__dict__, "resolved_llm_model", TestModel())


def _use_fake_food_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `Pipeline._ensure_food_db()` open `_FakeFoodDb` instead of the real artifact.

    For tests that go through the CLI (which constructs its own `Pipeline`
    with no way to inject a `food_db`), patching the `FoodDb` name that
    `pipeline.py` imported is the only seam available.
    """
    monkeypatch.setattr(pipeline_module, "FoodDb", _FakeFoodDb)


@pytest.mark.asyncio()
async def test_pipeline_baseline_produces_valid_meal_plan(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)

    baseline = VariantConfig(food_enabled=False, totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    with Pipeline(baseline, food_db=_FakeFoodDb()) as pipeline:  # type: ignore[arg-type]
        result = await pipeline.run(healthy_profile, "Plan one balanced day.")
    assert isinstance(result.plan, MealPlan)
    assert result.plan.user_id


@pytest.mark.asyncio()
async def test_pipeline_runs_with_food_db(
    monkeypatch: pytest.MonkeyPatch,
    vegan_peanut_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)

    variant = VariantConfig(rag_enabled=False, reflection_enabled=False)
    with Pipeline(variant, food_db=_FakeFoodDb()) as pipeline:  # type: ignore[arg-type]
        result = await pipeline.run(vegan_peanut_profile, "Plan a vegan day, no nuts.")
    assert isinstance(result.plan, MealPlan)
    assert result.shopping_list.items


@pytest.mark.asyncio()
async def test_pipeline_chat_threads_message_history(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)

    baseline = VariantConfig(food_enabled=False, totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    with Pipeline(baseline, food_db=_FakeFoodDb()) as pipeline:  # type: ignore[arg-type]
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


def test_compose_prompt_includes_meal_concepts_section(healthy_profile: UserProfile) -> None:
    with Pipeline(VariantConfig(rag_enabled=False), food_db=_FakeFoodDb()) as pipeline:  # type: ignore[arg-type]
        prompt = pipeline._compose_prompt(
            healthy_profile,
            "Plan a day.",
            healthy_profile.targets,
            [],
            [MealConcept(kind="breakfast", dish_name="Shakshuka with feta and crusty bread")],
        )
    assert "Meal concepts" in prompt
    assert "breakfast: Shakshuka with feta and crusty bread" in prompt


@pytest.mark.asyncio()
async def test_pipeline_survives_blueprint_agent_failure(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    """A broken brainstorming pass degrades to no concepts rather than failing the request."""
    _use_test_model(monkeypatch)

    def _boom() -> None:
        raise RuntimeError("blueprint agent unavailable")

    monkeypatch.setattr(pipeline_module, "build_blueprint_agent", _boom)

    baseline = VariantConfig(food_enabled=False, totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    with Pipeline(baseline, food_db=_FakeFoodDb()) as pipeline:  # type: ignore[arg-type]
        result = await pipeline.run(healthy_profile, "Plan one balanced day.")
    assert isinstance(result.plan, MealPlan)


def test_cli_chat_session_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_test_model(monkeypatch)
    _use_fake_food_db(monkeypatch)
    profile_id = next(iter(PROFILES))  # any built-in profile id

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["chat", "--profile-id", profile_id, "--no-off", "--no-totaller", "--no-rag", "--no-reflective-loop"],
        input="make breakfast lower-carb\nquit\n",
    )
    assert result.exit_code == 0, result.stdout
    assert "Shopping list" in result.stdout
