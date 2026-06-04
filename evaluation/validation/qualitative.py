"""Stage 3: LLM-as-judge (G-Eval) for soft session preferences."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from dietary_advisor.agents.prompts import JUDGE_SOFT_PREFERENCES_SYSTEM
from dietary_advisor.config import get_settings
from dietary_advisor.schemas.agent_output import AgentMealPlan
from evaluation.scenarios import SoftCriterion


class CriterionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class QualitativeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[CriterionScore] = Field(default_factory=list)
    aggregate: float = Field(ge=0.0, le=1.0)


def _build_judge_prompt(plan: AgentMealPlan, query: str, criteria: tuple[SoftCriterion, ...]) -> str:
    rubric = "\n".join(f"- {c.id}: {c.description}" for c in criteria)
    return (
        f"User query:\n{query}\n\n"
        f"Soft criteria to score:\n{rubric}\n\n"
        f"Meal plan (JSON):\n{plan.model_dump_json()}\n\n"
        "Score every criterion id listed above."
    )


async def score_soft_preferences(
    plan: AgentMealPlan,
    query: str,
    criteria: tuple[SoftCriterion, ...],
    *,
    model: str | Model | None = None,
) -> QualitativeResult | None:
    """Run a single G-Eval judge call; returns None when there are no soft criteria."""
    if not criteria:
        return None

    settings = get_settings()
    judge = Agent(
        model or settings.judge_model or settings.llm_model,
        output_type=QualitativeResult,
        system_prompt=JUDGE_SOFT_PREFERENCES_SYSTEM,
        retries=1,
    )
    prompt = _build_judge_prompt(plan, query, criteria)
    res = await judge.run(prompt)
    result = res.output
    if not result.scores:
        return QualitativeResult(scores=[], aggregate=0.0)
    if len(result.scores) != len(criteria):
        # Normalise aggregate over returned scores only.
        agg = sum(s.score for s in result.scores) / len(result.scores)
        return QualitativeResult(scores=result.scores, aggregate=round(agg, 4))
    expected_ids = {c.id for c in criteria}
    got_ids = {s.criterion_id for s in result.scores}
    if got_ids != expected_ids:
        agg = sum(s.score for s in result.scores) / len(result.scores)
        return QualitativeResult(scores=result.scores, aggregate=round(agg, 4))
    agg = sum(s.score for s in result.scores) / len(result.scores)
    return QualitativeResult(scores=result.scores, aggregate=round(agg, 4))
