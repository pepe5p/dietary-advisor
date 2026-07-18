"""Stage 3: LLM-as-judge (G-Eval) for soft session preferences."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from dietary_advisor.config import get_settings
from dietary_advisor.schemas.agent_output import AgentMealPlan
from evaluation.scenarios import SoftCriterion

JUDGE_SOFT_PREFERENCES_SYSTEM = """You are a G-Eval judge for dietary meal-plan quality. You score how well a
generated one-day meal plan satisfies *soft* session preferences from the user's
query. Focus on semantic fit to the stated soft criteria.

For each criterion you receive:
- Assign a score from 0.0 (not satisfied) to 1.0 (fully satisfied).
- Provide a one-sentence reasoning citing concrete plan elements (meal names,
  recipe instructions, ingredients).

Be strict but fair: partial satisfaction should score between 0.3 and 0.7.
Return structured scores for every criterion id listed in the prompt.
"""


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
) -> QualitativeResult | None:
    """Run a single G-Eval judge call; returns None when there are no soft criteria."""
    if not criteria:
        return None

    settings = get_settings()
    judge = Agent(
        settings.resolved_judge_model,
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
