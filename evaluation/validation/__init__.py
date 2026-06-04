"""Three-stage evaluation: quantitative, structural, and qualitative validation."""

from evaluation.validation.hydrate import (
    EvalPlanConversion,
    hydrate_meal_plan,
    meal_plan_to_eval_plan,
)
from evaluation.validation.qualitative import (
    CriterionScore,
    QualitativeResult,
    score_soft_preferences,
)
from evaluation.validation.quantitative import macro_errors, NutrientErrors
from evaluation.validation.structural import check_integrity, structural_csr

__all__ = [
    "CriterionScore",
    "EvalPlanConversion",
    "NutrientErrors",
    "QualitativeResult",
    "check_integrity",
    "hydrate_meal_plan",
    "macro_errors",
    "meal_plan_to_eval_plan",
    "score_soft_preferences",
    "structural_csr",
]
