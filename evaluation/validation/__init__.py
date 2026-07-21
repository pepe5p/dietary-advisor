"""Three-stage evaluation: quantitative, structural, and qualitative validation.

Also home to the code-based constraint Validator (`validator.py` + `rules.py`):
constraints are ground truth for scoring only, so the deterministic checker
that enforces them lives in the evaluation package, not in production.
"""

from dietary_advisor.planning.hydration import (
    hydrate_meal_plan,
    total_agent_meal_plan,
)
from evaluation.validation.qualitative import (
    CriterionScore,
    QualitativeResult,
    score_soft_preferences,
)
from evaluation.validation.quantitative import macro_errors, NutrientErrors
from evaluation.validation.rules import (
    HardRule,
    MaxNutrientRule,
    MinNutrientRule,
    rules_from_constraints,
)
from evaluation.validation.structural import check_integrity, structural_csr
from evaluation.validation.validator import validate_meal_plan

__all__ = [
    "CriterionScore",
    "HardRule",
    "MaxNutrientRule",
    "MinNutrientRule",
    "NutrientErrors",
    "QualitativeResult",
    "check_integrity",
    "hydrate_meal_plan",
    "macro_errors",
    "rules_from_constraints",
    "score_soft_preferences",
    "structural_csr",
    "total_agent_meal_plan",
    "validate_meal_plan",
]
