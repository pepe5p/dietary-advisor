"""Metric implementations for the ablation study.

All metrics are computed *after* the pipeline runs - they consume the
`PipelineResult` (plan + report + targets + constraints + citations) plus the
ground-truth `UserProfile` and produce numeric scores.

Metrics implemented:
  * HSR  - Hard Satisfaction Rate (already produced by the Validator).
  * SSR  - Soft Satisfaction Rate (preferred-foods coverage, dislike avoidance).
  * CSR  - Constraint Satisfaction Rate = w*HSR + (1-w)*SSR
  * MAE  - Mean Absolute Error against macro targets (kcal-normalised).
  * MSE  - Mean Squared Error against macro targets (kcal-normalised).
  * Faithfulness - fraction of rationale sentences that share substantive
    n-gram overlap with at least one cited chunk (token-overlap
    approximation; an LLM-as-judge variant is also exposed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dietary_advisor.schemas.constraints import HardConstraint, ValidationReport
from dietary_advisor.schemas.meal_plan import Citation, MealPlan
from dietary_advisor.schemas.nutrition import MacroTargets, NutrientName
from dietary_advisor.schemas.profile import UserProfile

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "and", "or", "in", "to", "for", "on", "with", "by", "as",
        "is", "are", "be", "this", "that", "these", "those", "from", "at", "it", "its",
        "have", "has", "had", "will", "should", "can", "may", "should",
    },
)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2 and t.lower() not in _STOPWORDS}


# ---------------------------------------------------------------------------- HSR / SSR / CSR


def hsr(report: ValidationReport, constraints: list[HardConstraint]) -> float:
    """Fraction of hard constraints with zero violations."""
    if not constraints:
        return 1.0
    violated = {(v.constraint.kind, v.constraint.target) for v in report.violations}
    passed = sum(1 for c in constraints if (c.kind, c.target) not in violated)
    return passed / len(constraints)


def ssr(plan: MealPlan, profile: UserProfile) -> float:
    """Soft Satisfaction Rate.

    We measure two things:
      1. coverage: fraction of `preferred_foods` that appear (substring match)
         in at least one portion's food.name across the plan.
      2. avoidance: fraction of `disliked_foods` that DO NOT appear.
    The SSR is the unweighted mean of (1) and (2). When neither set is
    declared, SSR defaults to 1.0 (vacuously satisfied).
    """
    food_names = " ".join(p.food.name.lower() for m in plan.meals for p in m.recipe.portions)
    preferred = profile.preferred_foods
    disliked = profile.disliked_foods

    if not preferred and not disliked:
        return 1.0

    parts: list[float] = []
    if preferred:
        hits = sum(1 for f in preferred if f.lower() in food_names)
        parts.append(hits / len(preferred))
    if disliked:
        misses = sum(1 for f in disliked if f.lower() not in food_names)
        parts.append(misses / len(disliked))
    return sum(parts) / len(parts)


def csr(report: ValidationReport, plan: MealPlan, profile: UserProfile,
        constraints: list[HardConstraint], hard_weight: float = 0.7) -> float:
    """Constraint Satisfaction Rate = hard_weight * HSR + (1-hard_weight) * SSR."""
    h = hsr(report, constraints)
    s = ssr(plan, profile)
    return round(hard_weight * h + (1.0 - hard_weight) * s, 4)


# ---------------------------------------------------------------------------- MAE / MSE


_MACRO_NUTRIENTS = (
    NutrientName.ENERGY_KCAL,
    NutrientName.PROTEIN_G,
    NutrientName.CARBS_G,
    NutrientName.FAT_G,
    NutrientName.FIBER_G,
)


@dataclass
class NutrientErrors:
    mae: float
    mse: float
    per_nutrient: dict[str, float]


def nutrient_errors(report: ValidationReport, targets: MacroTargets) -> NutrientErrors:
    """Compute MAE/MSE between actual totals and target macros, normalised to %."""
    target_d = targets.as_dict()
    per: dict[str, float] = {}
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    for n in _MACRO_NUTRIENTS:
        target = target_d.get(n)
        if target is None or target == 0:
            continue
        actual = float(report.totals.get(n, 0.0))
        rel = (actual - target) / target * 100.0  # percentage error
        per[n.value] = round(rel, 2)
        abs_errors.append(abs(rel))
        sq_errors.append(rel * rel)

    mae = round(sum(abs_errors) / len(abs_errors), 2) if abs_errors else 0.0
    mse = round(sum(sq_errors) / len(sq_errors), 2) if sq_errors else 0.0
    return NutrientErrors(mae=mae, mse=mse, per_nutrient=per)


# ---------------------------------------------------------------------------- Faithfulness


def _sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def faithfulness(plan: MealPlan, citations: list[Citation] | None = None,
                 *, min_overlap: int = 3) -> float:
    """Fraction of rationale sentences with sufficient lexical overlap to a citation.

    `min_overlap` is the minimum number of shared content tokens (after
    stop-word removal) required for a sentence to count as supported. This
    is a conservative heuristic; for the formal LLM-as-judge variant see
    :func:`faithfulness_llm`.
    """
    sources = citations if citations is not None else plan.citations
    if not plan.rationale:
        return 1.0
    if not sources:
        return 0.0

    source_tokens = [_tokens(c.snippet) for c in sources]
    if not any(source_tokens):
        return 0.0

    sents = _sentences(plan.rationale)
    if not sents:
        return 1.0
    supported = 0
    for sent in sents:
        st = _tokens(sent)
        if not st:
            supported += 1
            continue
        if any(len(st & cs) >= min_overlap for cs in source_tokens):
            supported += 1
    return round(supported / len(sents), 4)


# ---------------------------------------------------------------------------- LLM-as-judge


async def faithfulness_llm(plan: MealPlan, citations: list[Citation] | None = None,
                           model: str | None = None) -> float:
    """Optional LLM-as-judge variant of Faithfulness.

    Implemented lazily so the dependency on a live LLM is only paid when
    explicitly requested (via `--judge` in the runner).
    """
    from pydantic_ai import Agent  # local import to avoid hard dep at import time

    from dietary_advisor.agents.prompts import JUDGE_FAITHFULNESS_SYSTEM
    from dietary_advisor.config import get_settings

    sources = citations if citations is not None else plan.citations
    if not plan.rationale:
        return 1.0
    if not sources:
        return 0.0

    settings = get_settings()
    judge = Agent(
        model or settings.judge_model or settings.llm_model,
        output_type=bool,
        system_prompt=JUDGE_FAITHFULNESS_SYSTEM,
        retries=1,
    )

    citations_block = "\n".join(f"- [{c.source}] {c.snippet}" for c in sources)
    sents = _sentences(plan.rationale)
    if not sents:
        return 1.0
    supported = 0
    for sent in sents:
        prompt = f"Rationale sentence: {sent}\n\nCitations:\n{citations_block}\n\nIs the sentence fully supported?"
        try:
            res = await judge.run(prompt)
            if bool(res.output):
                supported += 1
        except Exception:  # noqa: BLE001
            continue
    return round(supported / len(sents), 4)
