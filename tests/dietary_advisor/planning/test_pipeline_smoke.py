"""End-to-end pipeline smoke test using pydantic-ai's `TestModel`.

`Settings.resolved_llm_model` and `resolved_meal_idea_llm_model` are cached
properties, so pre-seeding their caches with a `TestModel` makes every agent
factory pick it up. This makes the run fully offline (no LLM API key required)
yet still exercises the full structured-output pipeline (profile is never
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

import pytest
from pydantic_ai.models.test import TestModel

import dietary_advisor.planning.pipeline as pipeline_module
from dietary_advisor.config import get_settings
from dietary_advisor.food_db import OFFItem
from dietary_advisor.food_db.facade import BatchLookupResult, LookupQuery, LookupResult, OFFHit, QueryLookupResult
from dietary_advisor.planning.meal_plan import MealPlan
from dietary_advisor.planning.pipeline import Pipeline, VariantConfig
from dietary_advisor.profile import UserProfile


class _FakeFoodDb:
    """Permissive stand-in for `FoodDb`: resolves any code, always finds a hit.

    Also exposes an `open` classmethod so it can substitute for the `FoodDb`
    class itself, matching `FoodDb.open(settings)`.
    """

    @classmethod
    def open(cls, settings: object = None) -> _FakeFoodDb:  # noqa: ARG003
        return cls()

    async def lookup(self, queries: Sequence[LookupQuery]) -> BatchLookupResult:
        hit = OFFHit.model_validate(
            {
                "code": "test:1",
                "name": "Test food",
                "energy_kcal": 200.0,
                "protein_g": 10.0,
                "carbs_g": 20.0,
                "fat_g": 5.0,
            }
        )
        return BatchLookupResult(
            queries=[
                QueryLookupResult(
                    query=q.query,
                    results=LookupResult(open_food_facts=[hit], usda=[]),
                )
                for q in queries
            ]
        )

    def get_food(self, code: str) -> OFFItem:
        return OFFItem(code=code, product_name="Test food", energy_kcal_in_100g=200.0)

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeFoodDb:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _use_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-seed resolved model caches so real (API-backed) resolution never runs."""
    test_model = TestModel()
    settings = get_settings().__dict__
    monkeypatch.setitem(settings, "resolved_llm_model", test_model)
    monkeypatch.setitem(settings, "resolved_meal_idea_llm_model", test_model)


@pytest.mark.asyncio()
async def test_pipeline_baseline_produces_valid_meal_plan(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)

    baseline = VariantConfig(totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
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
async def test_pipeline_survives_meal_idea_agent_failure(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    """A broken brainstorming pass degrades to no concepts rather than failing the request."""
    _use_test_model(monkeypatch)

    def _boom() -> None:
        raise RuntimeError("meal-idea agent unavailable")

    monkeypatch.setattr(pipeline_module, "build_meal_idea_agent", _boom)

    baseline = VariantConfig(totaller_enabled=False, rag_enabled=False, reflection_enabled=False)
    with Pipeline(baseline, food_db=_FakeFoodDb()) as pipeline:  # type: ignore[arg-type]
        result = await pipeline.run(healthy_profile, "Plan one balanced day.")
    assert isinstance(result.plan, MealPlan)
