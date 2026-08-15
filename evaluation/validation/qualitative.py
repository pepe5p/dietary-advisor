"""Stage 3: LLM-as-judge (G-Eval) for soft preferences plus allergen/diet safety."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.agents.prompt_blocks import has_user_request
from evaluation.judges import Judge
from evaluation.scenarios import ALWAYS_SCORED_SOFT_CRITERIA, SoftCriterion

log = logging.getLogger(__name__)

_JUDGE_INTRO_WITH_QUERY = """You are a G-Eval judge for dietary meal-plan quality. You score how well a
generated one-day meal plan satisfies *soft* session preferences from the user's
query, AND you check hard safety fit for declared allergens and diet pattern."""

_JUDGE_INTRO_NO_QUERY = """You are a G-Eval judge for dietary meal-plan quality. You score how well a
generated one-day meal plan satisfies the supplied soft criteria and profile
preferences, AND you check hard safety fit for declared allergens and diet pattern."""

_JUDGE_BODY = """
Safety (always check):
- Allergens: flag any ingredient that conflicts with a declared allergen (judge
  from names, recipe text, and ingredients — there are no reliable allergen tags).
- Diet pattern: flag any ingredient that violates the stated diet pattern
  (e.g. meat/dairy in a vegan plan). Omnivore means no diet restriction.
Score safety as `safety_adherence` from 0.0 (clear violation) to 1.0 (fully safe).
List each concrete allergen/diet violation in `safety_violations` (empty when safe).

Micronutrients in soft criteria:
- Ingredients come from a food database with partial micronutrient coverage: many
  Open Food Facts products declare only macros, and several nutrients named in
  the criteria are not in the database at all. The plan carries no nutrient numbers.
- Treat mg/µg ranges in the criteria as direction, not thresholds to check. Judge
  whether the chosen foods and portions could plausibly reach the target, using
  your own knowledge of food composition.
- Never lower a score because a number is absent or unverifiable. Credit a rationale
  that names a gap food cannot close (e.g. recommends supplementation) over one
  that overclaims coverage.

For each soft criterion you receive:
- Assign a score from 0.0 (not satisfied) to 1.0 (fully satisfied).
- Provide a one-sentence reasoning citing concrete plan elements (meal names,
  recipe instructions, ingredients).

Be strict but fair: partial satisfaction should score between 0.3 and 0.7.
Return structured scores for every soft criterion id listed in the prompt,
plus the safety fields.
"""


def judge_soft_preferences_system(*, has_user_query: bool = True) -> str:
    intro = _JUDGE_INTRO_WITH_QUERY if has_user_query else _JUDGE_INTRO_NO_QUERY
    return intro + _JUDGE_BODY


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


def _compose_criteria(criteria: tuple[SoftCriterion, ...]) -> tuple[SoftCriterion, ...]:
    seen: set[str] = set()
    out: list[SoftCriterion] = []
    for criterion in (*ALWAYS_SCORED_SOFT_CRITERIA, *criteria):
        if criterion.id in seen:
            continue
        seen.add(criterion.id)
        out.append(criterion)
    return tuple(out)


def _build_judge_prompt(
    plan: AgentMealPlan,
    query: str,
    criteria: tuple[SoftCriterion, ...],
    *,
    allergens: Sequence[str],
    diet_pattern: str,
) -> str:
    rubric = "\n".join(f"- {c.id}: {c.description}" for c in criteria)
    allergen_list = ", ".join(allergens) if allergens else "(none)"
    parts: list[str] = []
    if has_user_request(query):
        parts.append(f"User query:\n{query}\n")
    parts.extend(
        [
            f"Profile allergens: {allergen_list}",
            f"Profile diet pattern: {diet_pattern or 'omnivore'}\n",
            f"Soft criteria to score:\n{rubric}\n",
            f"Meal plan (JSON):\n{plan.model_dump_json()}\n",
            "Score every soft criterion id listed above. Always fill `safety_adherence` and `safety_violations`.",
        ],
    )
    return "\n".join(parts)


def _normalize_soft_aggregate(result: QualitativeResult) -> QualitativeResult:
    soft_agg = sum(s.score for s in result.scores) / len(result.scores)
    return result.model_copy(update={"aggregate": round(soft_agg, 4)})


def _validate_criteria_coverage(
    expected: tuple[SoftCriterion, ...],
) -> Callable[[QualitativeResult], QualitativeResult]:
    expected_ids = [c.id for c in expected]

    def validate(result: QualitativeResult) -> QualitativeResult:
        returned = [s.criterion_id for s in result.scores]
        missing = [cid for cid in expected_ids if cid not in returned]
        unexpected = [cid for cid in returned if cid not in expected_ids]
        duplicated = sorted({cid for cid in returned if returned.count(cid) > 1})
        if not (missing or unexpected or duplicated):
            return result
        log.warning(
            "Judge returned %d/%d criteria (missing=%s unexpected=%s duplicated=%s)",
            len(set(returned)),
            len(expected_ids),
            missing,
            unexpected,
            duplicated,
        )
        raise ModelRetry(
            f"Score exactly the soft criterion ids listed in the prompt, one entry each. "
            f"Missing: {missing}. Unexpected: {unexpected}. Duplicated: {duplicated}."
        )

    return validate


async def score_soft_preferences(
    plan: AgentMealPlan,
    query: str,
    criteria: tuple[SoftCriterion, ...] = (),
    *,
    allergens: Sequence[str] = (),
    diet_pattern: str = "omnivore",
    judge: Judge,
) -> QualitativeResult:
    """Run a G-Eval judge call for soft criteria and allergen/diet safety.

    `ALWAYS_SCORED_SOFT_CRITERIA` are merged with the caller's criteria (deduped
    by id). `safety_adherence` / `safety_violations` are always populated.
    """
    composed = _compose_criteria(criteria)
    query_present = has_user_request(query)
    judge_agent = Agent(
        judge.resolved_model,
        output_type=QualitativeResult,
        system_prompt=judge_soft_preferences_system(has_user_query=query_present),
        retries=2,
    )
    judge_agent.output_validator(_validate_criteria_coverage(composed))
    prompt = _build_judge_prompt(
        plan,
        query,
        composed,
        allergens=allergens,
        diet_pattern=diet_pattern,
    )
    res = await judge_agent.run(prompt)
    return _normalize_soft_aggregate(res.output)
