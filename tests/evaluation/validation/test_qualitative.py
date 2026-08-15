"""Tests for G-Eval soft-preference + safety judge."""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from evaluation.judges import all_judges
from evaluation.scenarios import ALWAYS_SCORED_SOFT_CRITERIA, SoftCriterion
from evaluation.validation.qualitative import (
    _build_judge_prompt,
    CriterionScore,
    judge_soft_preferences_system,
    QualitativeResult,
    score_soft_preferences,
)
from tests.evaluation.conftest import agent_plan_rice_lunch


@pytest.mark.asyncio()
async def test_score_soft_preferences_with_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = agent_plan_rice_lunch()
    criteria = (SoftCriterion("recipe_simplicity", "Recipes should be quick and simple."),)
    expected = QualitativeResult(
        scores=[
            CriterionScore(
                criterion_id="recipe-makes-sense",
                score=1.0,
                reasoning="Rice lunch steps are coherent.",
            ),
            CriterionScore(
                criterion_id="recipe_simplicity",
                score=0.85,
                reasoning="Single-step rice cooking is simple.",
            ),
        ],
        aggregate=0.925,
        safety_adherence=1.0,
        safety_violations=[],
    )
    monkeypatch.setattr(
        "evaluation.judges.resolve_judge_model",
        lambda _model_id: TestModel(custom_output_args=expected.model_dump()),
    )
    result = await score_soft_preferences(
        plan,
        "quick simple meals please",
        criteria,
        allergens=["peanuts"],
        diet_pattern="vegetarian",
        judge=all_judges()[0],
    )
    assert result is not None
    assert result.aggregate == pytest.approx(0.925)
    assert {s.criterion_id for s in result.scores} == {"recipe-makes-sense", "recipe_simplicity"}
    assert result.safety_adherence == pytest.approx(1.0)


@pytest.mark.asyncio()
async def test_score_soft_preferences_runs_safety_without_scenario_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = agent_plan_rice_lunch()
    expected = QualitativeResult(
        scores=[
            CriterionScore(
                criterion_id="recipe-makes-sense",
                score=0.9,
                reasoning="Rice lunch steps are coherent.",
            ),
        ],
        aggregate=0.9,
        safety_adherence=0.0,
        safety_violations=["Lunch uses milk yogurt but profile is vegan"],
    )
    monkeypatch.setattr(
        "evaluation.judges.resolve_judge_model",
        lambda _model_id: TestModel(custom_output_args=expected.model_dump()),
    )
    result = await score_soft_preferences(
        plan,
        "any query",
        (),
        allergens=[],
        diet_pattern="vegan",
        judge=all_judges()[0],
    )
    assert result is not None
    assert len(result.scores) == 1
    assert result.scores[0].criterion_id == "recipe-makes-sense"
    assert result.aggregate == pytest.approx(0.9)
    assert result.safety_adherence == pytest.approx(0.0)
    assert result.safety_violations


def test_judge_prompt_omits_user_query_when_blank() -> None:
    plan = agent_plan_rice_lunch()
    prompt = _build_judge_prompt(
        plan,
        "",
        ALWAYS_SCORED_SOFT_CRITERIA,
        allergens=[],
        diet_pattern="omnivore",
    )
    assert "User query:" not in prompt
    assert prompt.startswith("Profile allergens:")


def test_judge_system_prompt_omits_query_framing_when_absent() -> None:
    prompt = judge_soft_preferences_system(has_user_query=False)
    assert "session preferences from the user's query" not in prompt
    assert "supplied soft criteria" in prompt


def test_registered_judges_differ_only_by_model() -> None:
    judges = all_judges()

    assert len(judges) > 1
    assert len({judge.model_id for judge in judges}) == len(judges)
    assert len({judge.scores_subdir for judge in judges}) == len(judges)


@pytest.mark.asyncio()
async def test_score_soft_preferences_rejects_missing_criterion_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = agent_plan_rice_lunch()
    criteria = (SoftCriterion("recipe_simplicity", "Recipes should be quick and simple."),)
    partial = QualitativeResult(
        scores=[
            CriterionScore(
                criterion_id="recipe-makes-sense",
                score=1.0,
                reasoning="Rice lunch steps are coherent.",
            ),
        ],
        aggregate=1.0,
        safety_adherence=1.0,
        safety_violations=[],
    )
    monkeypatch.setattr(
        "evaluation.judges.resolve_judge_model",
        lambda _model_id: TestModel(custom_output_args=partial.model_dump()),
    )
    with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum retries"):
        await score_soft_preferences(
            plan,
            "quick simple meals please",
            criteria,
            judge=all_judges()[0],
        )


@pytest.mark.asyncio()
async def test_score_soft_preferences_rejects_unexpected_criterion_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = agent_plan_rice_lunch()
    unexpected = QualitativeResult(
        scores=[
            CriterionScore(
                criterion_id="recipe-makes-sense",
                score=1.0,
                reasoning="Rice lunch steps are coherent.",
            ),
            CriterionScore(
                criterion_id="not-in-rubric",
                score=0.5,
                reasoning="Extra criterion.",
            ),
        ],
        aggregate=0.75,
        safety_adherence=1.0,
        safety_violations=[],
    )
    monkeypatch.setattr(
        "evaluation.judges.resolve_judge_model",
        lambda _model_id: TestModel(custom_output_args=unexpected.model_dump()),
    )
    with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum retries"):
        await score_soft_preferences(
            plan,
            "",
            (),
            judge=all_judges()[0],
        )
