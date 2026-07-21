"""Stage 3: LLM-as-judge (G-Eval) for soft preferences plus allergen/diet safety."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from dietary_advisor.agents.agent_output import AgentMealPlan
from evaluation.scenarios import SoftCriterion
from evaluation.settings import get_evaluation_settings

JUDGE_SOFT_PREFERENCES_SYSTEM = """You are a G-Eval judge for dietary meal-plan quality. You score how well a
generated one-day meal plan satisfies *soft* session preferences from the user's
query, AND you check hard safety fit for declared allergens and diet pattern.

Safety (always check, even when there are no soft criteria):
- Allergens: flag any ingredient that conflicts with a declared allergen (judge
  from names, recipe text, and ingredients — there are no reliable allergen tags).
- Diet pattern: flag any ingredient that violates the stated diet pattern
  (e.g. meat/dairy in a vegan plan). Omnivore means no diet restriction.
Score safety as `safety_adherence` from 0.0 (clear violation) to 1.0 (fully safe).
List each concrete allergen/diet violation in `safety_violations` (empty when safe).

For each soft criterion you receive:
- Assign a score from 0.0 (not satisfied) to 1.0 (fully satisfied).
- Provide a one-sentence reasoning citing concrete plan elements (meal names,
  recipe instructions, ingredients).

Be strict but fair: partial satisfaction should score between 0.3 and 0.7.
Return structured scores for every soft criterion id listed in the prompt
(an empty list when none were supplied), plus the safety fields.
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
    safety_adherence: float = Field(ge=0.0, le=1.0, default=1.0)
    safety_violations: list[str] = Field(default_factory=list)


def _build_judge_prompt(
    plan: AgentMealPlan,
    query: str,
    criteria: tuple[SoftCriterion, ...],
    *,
    allergens: Sequence[str],
    diet_pattern: str,
) -> str:
    if criteria:
        rubric = "\n".join(f"- {c.id}: {c.description}" for c in criteria)
        soft_block = f"Soft criteria to score:\n{rubric}\n\n"
        soft_tail = "Score every soft criterion id listed above."
    else:
        soft_block = "Soft criteria to score: (none)\n\n"
        soft_tail = "Return an empty `scores` list for soft criteria."
    allergen_list = ", ".join(allergens) if allergens else "(none)"
    return (
        f"User query:\n{query}\n\n"
        f"Profile allergens: {allergen_list}\n"
        f"Profile diet pattern: {diet_pattern or 'omnivore'}\n\n"
        f"{soft_block}"
        f"Meal plan (JSON):\n{plan.model_dump_json()}\n\n"
        f"{soft_tail} Always fill `safety_adherence` and `safety_violations`."
    )


def _normalize_soft_aggregate(result: QualitativeResult, criteria: tuple[SoftCriterion, ...]) -> QualitativeResult:
    if not result.scores:
        soft_agg = 1.0 if not criteria else 0.0
    else:
        soft_agg = sum(s.score for s in result.scores) / len(result.scores)
    return result.model_copy(update={"aggregate": round(soft_agg, 4)})


async def score_soft_preferences(
    plan: AgentMealPlan,
    query: str,
    criteria: tuple[SoftCriterion, ...] = (),
    *,
    allergens: Sequence[str] = (),
    diet_pattern: str = "omnivore",
) -> QualitativeResult:
    """Run a G-Eval judge call for soft criteria and allergen/diet safety.

    Soft criteria may be empty; `safety_adherence` / `safety_violations` are
    always populated. The caller decides whether to invoke this at all.
    """
    settings = get_evaluation_settings()
    judge = Agent(
        settings.resolved_judge_model,
        output_type=QualitativeResult,
        system_prompt=JUDGE_SOFT_PREFERENCES_SYSTEM,
        retries=1,
    )
    prompt = _build_judge_prompt(
        plan,
        query,
        criteria,
        allergens=allergens,
        diet_pattern=diet_pattern,
    )
    res = await judge.run(prompt)
    return _normalize_soft_aggregate(res.output, criteria)
