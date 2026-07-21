"""Offline tests for the critique-then-refine reflection loop.

`run_agent_logged` is monkeypatched with a scripted stand-in (rather than
seeding `TestModel` output per call), since the loop drives two structurally
different agents (a `PlanCritique` critic and an `AgentMealPlan` refiner) and
needs full control over which one "finds" issues on which iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

import dietary_advisor.reflection.reflection as reflection_module
from dietary_advisor.agents.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.reflection import PlanCritique
from dietary_advisor.config import get_settings
from dietary_advisor.food_db import OFFItem
from dietary_advisor.profile import UserProfile
from dietary_advisor.reflection import reflect_and_refine
from dietary_advisor.totaller.nutrition import NutrientName
from tests.conftest import LONG_INSTRUCTIONS

_KNOWN_CODE = "known"


class _FakeFoodDb:
    """Resolves exactly one code, with fixed per-100g nutrients."""

    def get_food(self, code: str) -> OFFItem:
        assert code == _KNOWN_CODE
        return OFFItem(
            code=code,
            name="Test food",
            nutrients_per_100g={
                NutrientName.ENERGY_KCAL: 200.0,
                NutrientName.PROTEIN_G: 10.0,
                NutrientName.CARBS_G: 20.0,
                NutrientName.FAT_G: 5.0,
            },
        )


@dataclass
class _StubResult:
    """Minimal stand-in for `AgentRunResult` satisfying `collect_from_result`'s protocol."""

    output: Any
    _messages: list[ModelMessage] = field(default_factory=list)

    def all_messages(self) -> list[ModelMessage]:
        return self._messages

    def usage(self) -> RunUsage:
        return RunUsage()


def _plan(name: str = "Dish") -> AgentMealPlan:
    return AgentMealPlan(
        user_id="t",
        meals=[
            AgentMeal(
                kind="lunch",
                recipe=AgentRecipe(
                    name=name,
                    portions=[PortionRef(code=_KNOWN_CODE, name="Test food", grams=100.0)],
                    instructions=LONG_INSTRUCTIONS,
                ),
            ),
        ],
    )


def _deps(profile: UserProfile) -> AgentDeps:
    return AgentDeps(profile=profile, targets=profile.targets, food_db=_FakeFoodDb())  # type: ignore[arg-type]


def _use_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-seed `resolved_llm_model` so building the critic/refiner agents never resolves a real model."""
    monkeypatch.setitem(get_settings().__dict__, "resolved_llm_model", TestModel())


def _install_scripted_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    critic_outputs: list[PlanCritique],
    refiner_outputs: list[AgentMealPlan | Exception] | None = None,
) -> list[tuple[str, str]]:
    """Patch `run_agent_logged` to serve scripted per-role outputs, recording each (label, prompt) call."""
    calls: list[tuple[str, str]] = []
    critic_queue = list(critic_outputs)
    refiner_queue = list(refiner_outputs or [])

    async def _fake_run_agent_logged(
        agent: object,  # noqa: ARG001
        prompt: str,
        *,
        deps: object,  # noqa: ARG001
        label: str,
    ) -> _StubResult:
        calls.append((label, prompt))
        if label.startswith("critic"):
            return _StubResult(output=critic_queue.pop(0))
        if label.startswith("refiner"):
            outcome = refiner_queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return _StubResult(output=outcome)
        raise AssertionError(f"unexpected label: {label}")

    monkeypatch.setattr(reflection_module, "run_agent_logged", _fake_run_agent_logged)
    return calls


@pytest.mark.asyncio()
async def test_critic_approval_skips_refiner(monkeypatch: pytest.MonkeyPatch, healthy_profile: UserProfile) -> None:
    _use_test_model(monkeypatch)
    calls = _install_scripted_runner(monkeypatch, critic_outputs=[PlanCritique(issues=[])])
    plan = _plan()

    result = await reflect_and_refine(plan, _deps(healthy_profile), "Plan a day.", totaller_enabled=True)

    assert result.plan is plan
    assert result.iterations == 0
    assert result.approved is True
    assert [label for label, _ in calls] == ["critic#1"]


@pytest.mark.asyncio()
async def test_critic_issues_trigger_one_refiner_pass(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)
    revised = _plan("Revised dish")
    calls = _install_scripted_runner(
        monkeypatch,
        critic_outputs=[
            PlanCritique(issues=["Lunch uses feta cheese but the profile is vegan"]),
            PlanCritique(issues=[]),
        ],
        refiner_outputs=[revised],
    )

    result = await reflect_and_refine(_plan(), _deps(healthy_profile), "Plan a day.", totaller_enabled=True)

    assert result.plan is revised
    assert result.iterations == 1
    assert result.approved is True
    assert [label for label, _ in calls] == ["critic#1", "refiner#1", "critic#2"]
    refiner_prompt = calls[1][1]
    assert "Lunch uses feta cheese but the profile is vegan" in refiner_prompt


@pytest.mark.asyncio()
async def test_totaller_disabled_omits_totals_feedback(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)
    calls = _install_scripted_runner(monkeypatch, critic_outputs=[PlanCritique(issues=[])])

    await reflect_and_refine(_plan(), _deps(healthy_profile), "Plan a day.", totaller_enabled=False)

    assert "Deterministic nutrient totals" not in calls[0][1]


@pytest.mark.asyncio()
async def test_totaller_enabled_includes_totals_feedback(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)
    calls = _install_scripted_runner(monkeypatch, critic_outputs=[PlanCritique(issues=[])])

    await reflect_and_refine(_plan(), _deps(healthy_profile), "Plan a day.", totaller_enabled=True)

    critic_prompt = calls[0][1]
    assert "Deterministic nutrient totals" in critic_prompt
    assert "200 kcal" in critic_prompt  # 100g portion @ 200 kcal/100g


@pytest.mark.asyncio()
async def test_refiner_failure_keeps_previous_plan(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)
    plan = _plan()
    _install_scripted_runner(
        monkeypatch,
        critic_outputs=[PlanCritique(issues=["Some issue"])],
        refiner_outputs=[RuntimeError("refiner exploded")],
    )

    result = await reflect_and_refine(plan, _deps(healthy_profile), "Plan a day.", totaller_enabled=True)

    assert result.plan is plan
    assert result.iterations == 0
    assert result.approved is False


@pytest.mark.asyncio()
async def test_budget_exhausted_returns_last_refined_plan(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)
    revised = _plan("Still flawed")
    _install_scripted_runner(
        monkeypatch,
        critic_outputs=[
            PlanCritique(issues=["issue A"]),
            PlanCritique(issues=["issue B"]),
        ],
        refiner_outputs=[revised, revised],
    )

    result = await reflect_and_refine(
        _plan(),
        _deps(healthy_profile),
        "Plan a day.",
        max_loops=2,
        totaller_enabled=True,
    )

    assert result.plan is revised
    assert result.iterations == 2
    assert result.approved is False


@pytest.mark.asyncio()
async def test_zero_budget_skips_loop_entirely(
    monkeypatch: pytest.MonkeyPatch,
    healthy_profile: UserProfile,
) -> None:
    _use_test_model(monkeypatch)
    calls = _install_scripted_runner(monkeypatch, critic_outputs=[])
    plan = _plan()

    result = await reflect_and_refine(plan, _deps(healthy_profile), "Plan a day.", max_loops=0)

    assert result.plan is plan
    assert result.iterations == 0
    assert result.approved is False
    assert calls == []
