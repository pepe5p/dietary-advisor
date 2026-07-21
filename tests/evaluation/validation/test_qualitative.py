"""Tests for G-Eval soft-preference judge."""

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
    )
    assert result is not None
    assert result.aggregate == pytest.approx(0.85)
    assert result.scores[0].criterion_id == "recipe_simplicity"


@pytest.mark.asyncio()
async def test_score_soft_preferences_none_when_no_criteria() -> None:
    plan = agent_plan_rice_lunch()
    result = await score_soft_preferences(plan, "any query", ())
    assert result is None
