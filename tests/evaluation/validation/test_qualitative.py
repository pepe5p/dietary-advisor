"""Tests for G-Eval soft-preference + safety judge."""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from evaluation.scenarios import SoftCriterion
from evaluation.settings import get_evaluation_settings
from evaluation.validation.qualitative import CriterionScore, QualitativeResult, score_soft_preferences
from tests.evaluation.conftest import agent_plan_rice_lunch


@pytest.mark.asyncio()
async def test_score_soft_preferences_with_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = agent_plan_rice_lunch()
    criteria = (SoftCriterion("recipe_simplicity", "Recipes should be quick and simple."),)
    expected = QualitativeResult(
        scores=[
            CriterionScore(
                criterion_id="recipe_simplicity",
                score=0.85,
                reasoning="Single-step rice cooking is simple.",
            ),
        ],
        aggregate=0.85,
        safety_adherence=1.0,
        safety_violations=[],
    )
    monkeypatch.setitem(
        get_evaluation_settings().__dict__,
        "resolved_judge_model",
        TestModel(custom_output_args=expected.model_dump()),
    )
    result = await score_soft_preferences(
        plan,
        "quick simple meals please",
        criteria,
        allergens=["peanuts"],
        diet_pattern="vegetarian",
    )
    assert result is not None
    assert result.aggregate == pytest.approx(0.85)
    assert result.scores[0].criterion_id == "recipe_simplicity"
    assert result.safety_adherence == pytest.approx(1.0)


@pytest.mark.asyncio()
async def test_score_soft_preferences_runs_safety_without_soft_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = agent_plan_rice_lunch()
    expected = QualitativeResult(
        scores=[],
        aggregate=1.0,
        safety_adherence=0.0,
        safety_violations=["Lunch uses milk yogurt but profile is vegan"],
    )
    monkeypatch.setitem(
        get_evaluation_settings().__dict__,
        "resolved_judge_model",
        TestModel(custom_output_args=expected.model_dump()),
    )
    result = await score_soft_preferences(
        plan,
        "any query",
        (),
        allergens=[],
        diet_pattern="vegan",
    )
    assert result is not None
    assert result.scores == []
    assert result.aggregate == pytest.approx(1.0)
    assert result.safety_adherence == pytest.approx(0.0)
    assert result.safety_violations
