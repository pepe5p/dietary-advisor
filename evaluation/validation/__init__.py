"""Two-stage evaluation: quantitative macro error and qualitative soft-preference judge."""

from evaluation.validation.qualitative import (
    CriterionScore,
    judge_soft_preferences_system,
    QualitativeResult,
    score_soft_preferences,
)
from evaluation.validation.quantitative import macro_errors, NutrientErrors

__all__ = [
    "CriterionScore",
    "NutrientErrors",
    "QualitativeResult",
    "judge_soft_preferences_system",
    "macro_errors",
    "score_soft_preferences",
]
