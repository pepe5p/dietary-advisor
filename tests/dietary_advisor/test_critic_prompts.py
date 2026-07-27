"""Critic agent prompt composition (`dietary_advisor.agents.critic.prompts`)."""

from __future__ import annotations

import pytest

from dietary_advisor.agents.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.agents.critic.prompts import critic_agent_system, critic_user_prompt
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.profile import UserProfile
from tests.conftest import LONG_INSTRUCTIONS, LONG_RATIONALE


@pytest.fixture()
def deps(healthy_profile: UserProfile) -> AgentDeps:
    return AgentDeps(profile=healthy_profile, targets=healthy_profile.targets, food_db=None)  # type: ignore[arg-type]


def _plan() -> AgentMealPlan:
    return AgentMealPlan(
        user_id="t",
        meals=[
            AgentMeal(
                kind="lunch",
                recipe=AgentRecipe(
                    name="Dish",
                    portions=[PortionRef(code="off:1", name="Test food", grams=100.0)],
                    instructions=LONG_INSTRUCTIONS,
                ),
            ),
        ],
        rationale=LONG_RATIONALE,
    )


def test_critic_system_prompt_omits_request_check_when_absent() -> None:
    prompt = critic_agent_system(has_user_request=False)
    assert "original request" not in prompt
    assert "Do NOT flag a deviation the user explicitly requested" not in prompt
    assert "1. Allergens:" in prompt
    assert "2. Diet pattern:" in prompt
    assert "3. Disliked and preferred foods" in prompt
    assert "4. Macro totals vs targets." in prompt


def test_critic_system_prompt_includes_request_check_when_present() -> None:
    prompt = critic_agent_system(has_user_request=True)
    assert "The user's original request:" in prompt
    assert "deviation the user explicitly requested" in prompt


@pytest.mark.parametrize("user_query", ["", "   "])
def test_critic_user_prompt_omits_request_when_blank(deps: AgentDeps, user_query: str) -> None:
    prompt = critic_user_prompt(_plan(), deps, user_query, totals_feedback=None, rag_citations=[])
    assert "User request:" not in prompt
    assert prompt.startswith("Profile:")


def test_critic_user_prompt_includes_request_when_present(deps: AgentDeps) -> None:
    prompt = critic_user_prompt(
        _plan(),
        deps,
        "I want pizza.",
        totals_feedback=None,
        rag_citations=[],
    )
    assert "User request: I want pizza." in prompt
    assert LONG_INSTRUCTIONS in prompt
    assert LONG_RATIONALE in prompt
