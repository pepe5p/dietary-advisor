"""Derive ground-truth hard constraints from a `UserProfile`.

This is the bridge from a user-facing profile to the machine-readable rules
consumed by the evaluation Validator. It encodes the safety floors mentioned
in the literature (e.g. WHO sodium cap, ADA fiber floor for type-2 diabetes).

Evaluation-only: production never derives or sees constraints - the agent
must infer restrictions from the profile itself. This function exists so the
frozen `hard_constraints` in `evaluation/profiles/cases.py` can be verified
against the profile they were authored from (see `test_cases.py`).

Allergen and diet-pattern adherence are deliberately *not* derived here: they
are scored by the LLM critic / qualitative G-Eval judge rather than
deterministic tag matching.
"""

from __future__ import annotations

from dietary_advisor.profile import UserProfile
from dietary_advisor.totaller.nutrition import NutrientName
from evaluation.constraints import ConstraintSource, HardConstraint
from evaluation.profiles.vocab import Condition

# Clinical safety constants. These are conservative defaults sourced from the
# literature in the kwerenda; they are *not* clinical guidelines themselves
# and must be cross-validated by the RAG layer before patient-facing use.
_CONDITION_RULES: dict[str, list[HardConstraint]] = {
    Condition.HYPERTENSION.value: [
        HardConstraint.max_nutrient(
            NutrientName.SODIUM_MG,
            value=2000.0,  # WHO recommendation for adults with hypertension
            rationale="WHO recommends <2 g/day sodium for adults with hypertension.",
        ),
    ],
    Condition.TYPE_2_DIABETES.value: [
        HardConstraint.min_nutrient(
            NutrientName.FIBER_G,
            value=30.0,
            rationale="ADA Standards of Care: target >=30 g fiber/day to improve glycaemic control.",
        ),
        HardConstraint.max_nutrient(
            NutrientName.SUGAR_G,
            value=50.0,
            rationale="WHO conditional recommendation: <10 % of energy from free sugars (~50 g on a 2000 kcal diet).",
        ),
    ],
    Condition.DYSLIPIDEMIA.value: [
        HardConstraint.max_nutrient(
            NutrientName.SATURATED_FAT_G,
            value=20.0,
            rationale="ESC/EAS lipid guideline: limit saturated fat to <10% of energy.",
        ),
        HardConstraint.max_nutrient(
            NutrientName.CHOLESTEROL_MG,
            value=300.0,
            rationale="Common dyslipidaemia threshold: <300 mg dietary cholesterol/day.",
        ),
    ],
    Condition.CKD_STAGE_3.value: [
        HardConstraint.max_nutrient(
            NutrientName.SODIUM_MG,
            value=2000.0,
            rationale="KDOQI: <2 g/day sodium for CKD stage 3+.",
        ),
        HardConstraint.max_nutrient(
            NutrientName.POTASSIUM_MG,
            value=2400.0,
            rationale="KDOQI: <2.4 g/day potassium for CKD stage 3+.",
        ),
    ],
}


def derive_hard_constraints(profile: UserProfile) -> list[HardConstraint]:
    """Enumerate deterministic hard constraints implied by the profile.

    Combines ingredient exclusions (disliked foods) with clinical-condition
    nutrient rules from `_CONDITION_RULES`. Allergen / diet-pattern rules are
    left to the LLM judge.
    """
    out: list[HardConstraint] = []
    for disliked in profile.disliked_foods:
        out.append(
            HardConstraint(
                kind="ingredient_exclusion",
                target=disliked,
                source=ConstraintSource.PROFILE,
            ),
        )
    for cond in profile.conditions:
        out.extend(_CONDITION_RULES.get(cond, []))
    return out
