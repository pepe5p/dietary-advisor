"""User-turn prompt composition (`dietary_advisor.agents.nutrition.prompts`)."""

from __future__ import annotations

import pytest

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.meal_idea import MealConcept
from dietary_advisor.agents.nutrition.prompts import nutrition_agent_system, nutrition_user_prompt
from dietary_advisor.profile import UserProfile


@pytest.fixture()
def deps(healthy_profile: UserProfile) -> AgentDeps:
    # The prompt builders only read `profile`/`targets`; the food DB is never touched.
    return AgentDeps(profile=healthy_profile, targets=healthy_profile.targets, food_db=None)  # type: ignore[arg-type]


def test_nutrition_prompt_includes_meal_concepts_section(deps: AgentDeps) -> None:
    prompt = nutrition_user_prompt(
        deps,
        "Plan a day.",
        rag_citations=[],
        meal_concepts=[MealConcept(kind="breakfast", dish_name="Shakshuka with feta and crusty bread")],
    )
    assert "Meal concepts" in prompt
    assert "breakfast: Shakshuka with feta and crusty bread" in prompt


def test_nutrition_prompt_groups_concepts_by_slot(deps: AgentDeps) -> None:
    prompt = nutrition_user_prompt(
        deps,
        "Plan a day.",
        rag_citations=[],
        meal_concepts=[
            MealConcept(kind="breakfast", dish_name="Shakshuka"),
            MealConcept(kind="lunch", dish_name="Ramen"),
            MealConcept(kind="breakfast", dish_name="Menemen"),
        ],
    )
    assert "- breakfast: Shakshuka | Menemen" in prompt
    assert "- lunch: Ramen" in prompt


def test_nutrition_prompt_omits_concepts_section_when_empty(deps: AgentDeps) -> None:
    prompt = nutrition_user_prompt(deps, "Plan a day.", rag_citations=[], meal_concepts=[])
    assert "Meal concepts" not in prompt


@pytest.mark.parametrize(
    ("totaller_enabled", "rag_enabled"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_nutrition_system_prompt_includes_implicit_requirements(
    totaller_enabled: bool,
    rag_enabled: bool,
) -> None:
    prompt = nutrition_agent_system(totaller_enabled=totaller_enabled, rag_enabled=rag_enabled)
    assert "Implicit requirements:" in prompt


def test_nutrition_system_prompt_always_includes_rationale_rule() -> None:
    prompt = nutrition_agent_system(rag_enabled=False, totaller_enabled=False)
    assert "Always fill `rationale`" in prompt
    assert "supplementation" in prompt.lower()
    assert "Cite every clinical claim" not in prompt


def test_nutrition_system_prompt_includes_citation_rule_when_rag_enabled() -> None:
    prompt = nutrition_agent_system(rag_enabled=True, totaller_enabled=False)
    assert "Always fill `rationale`" in prompt
    assert "Cite every clinical claim" in prompt


@pytest.mark.parametrize("user_query", ["", "   "])
def test_nutrition_user_prompt_omits_request_when_blank(deps: AgentDeps, user_query: str) -> None:
    prompt = nutrition_user_prompt(deps, user_query, rag_citations=[], meal_concepts=[])
    assert "User request:" not in prompt
    assert prompt.startswith("Profile:")


def test_nutrition_user_prompt_includes_request_when_present(deps: AgentDeps) -> None:
    prompt = nutrition_user_prompt(deps, "I want pizza.", rag_citations=[], meal_concepts=[])
    assert "User request: I want pizza." in prompt


def test_nutrition_system_prompt_includes_request_precedence_when_present() -> None:
    prompt = nutrition_agent_system(has_user_request=True)
    assert "explicit user request beats these guardrails" in prompt


def test_nutrition_system_prompt_omits_request_precedence_when_absent() -> None:
    prompt = nutrition_agent_system(has_user_request=False)
    assert "explicit user request beats these guardrails" not in prompt
    assert "profile conditions and supplied clinical-guideline excerpts" in prompt.lower()
