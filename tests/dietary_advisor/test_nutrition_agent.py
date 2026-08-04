"""The nutrition/refiner agents' toolset and the code output-validator."""

from __future__ import annotations

import pytest
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from dietary_advisor.agents.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition.agent import (
    _validate_codes,
    build_nutrition_agent,
    build_refiner_agent,
    total_meal_plan,
)
from dietary_advisor.config import get_settings
from dietary_advisor.food_db import OFFItem, OFFUnknownFoodCodeError
from dietary_advisor.profile import UserProfile
from dietary_advisor.totaller.nutrition import MacroTargets, NutrientName
from tests.conftest import LONG_INSTRUCTIONS, LONG_RATIONALE

_ALWAYS_ON_TOOLS = {"lookup_foods", "total_meal_plan"}


def _tool_names(agent: object) -> set[str]:
    return set(agent._function_toolset.tools)  # type: ignore[attr-defined]


def test_nutrition_and_refiner_agents_always_register_lookup_and_totaller_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(get_settings().__dict__, "resolved_llm_model", TestModel())

    nutrition = build_nutrition_agent(totaller_enabled=True)
    refiner = build_refiner_agent(totaller_enabled=True)

    assert _tool_names(nutrition) >= _ALWAYS_ON_TOOLS
    assert _tool_names(refiner) >= _ALWAYS_ON_TOOLS


def test_totaller_tool_omitted_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(get_settings().__dict__, "resolved_llm_model", TestModel())

    agent = build_nutrition_agent(totaller_enabled=False)
    assert "total_meal_plan" not in _tool_names(agent)
    assert "lookup_foods" in _tool_names(agent)


class _FakeFoodDb:
    """Knows exactly one code; everything else raises `UnknownFoodCodeError` like the real facade."""

    def __init__(self, known_code: str = "known") -> None:
        self._known_code = known_code

    def get_food(self, code: str) -> OFFItem:
        if code != self._known_code:
            raise OFFUnknownFoodCodeError(code, f"unknown code: {code!r}")
        return OFFItem(code=code, product_name="Known food", energy_kcal_in_100g=100.0)


def _fake_ctx(food_db: object) -> RunContext[AgentDeps]:
    targets = MacroTargets(energy_kcal=2000.0, protein_g=100.0, carbs_g=250.0, fat_g=60.0)
    deps = AgentDeps(
        profile=UserProfile(user_id="t", age=30, sex="male", height_cm=180, weight_kg=80, targets=targets),
        targets=targets,
        food_db=food_db,  # type: ignore[arg-type]
    )
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def _plan_with_code(code: str) -> AgentMealPlan:
    return AgentMealPlan(
        user_id="t",
        meals=[
            AgentMeal(
                kind="lunch",
                recipe=AgentRecipe(
                    name="Dish",
                    portions=[PortionRef(code=code, name="Known food", grams=100.0)],
                    instructions=LONG_INSTRUCTIONS,
                ),
            ),
        ],
        rationale=LONG_RATIONALE,
    )


@pytest.mark.asyncio()
async def test_validate_codes_passes_known_codes() -> None:
    ctx = _fake_ctx(_FakeFoodDb("known"))
    plan = _plan_with_code("known")
    assert await _validate_codes(ctx, plan) is plan


@pytest.mark.asyncio()
async def test_validate_codes_retries_on_unknown_code() -> None:
    ctx = _fake_ctx(_FakeFoodDb("known"))
    plan = _plan_with_code("made-up-code")
    with pytest.raises(ModelRetry):
        await _validate_codes(ctx, plan)


@pytest.mark.asyncio()
async def test_total_meal_plan_tool_hydrates_and_sums() -> None:
    ctx = _fake_ctx(_FakeFoodDb("known"))
    plan = _plan_with_code("known")
    totals = await total_meal_plan(ctx, plan)
    assert totals.totals[NutrientName.ENERGY_KCAL] == pytest.approx(100.0)  # 100g @ 100 kcal/100g
    assert len(totals.per_meal) == 1
    assert totals.per_meal[0].totals[NutrientName.ENERGY_KCAL] == pytest.approx(100.0)


@pytest.mark.asyncio()
async def test_total_meal_plan_tool_retries_on_unknown_code() -> None:
    ctx = _fake_ctx(_FakeFoodDb("known"))
    plan = _plan_with_code("made-up-code")
    with pytest.raises(ModelRetry):
        await total_meal_plan(ctx, plan)
