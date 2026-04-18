"""Pydantic-based DSL: the symbolic interface between the LLM and the deterministic core."""

from dietary_advisor.schemas.constraints import (
    HardConstraint,
    SoftConstraint,
    ValidationReport,
    Violation,
)
from dietary_advisor.schemas.meal_plan import (
    Citation,
    Meal,
    MealKind,
    MealPlan,
    NutrientTotals,
    Portion,
    Recipe,
)
from dietary_advisor.schemas.nutrition import (
    FoodItem,
    MacroTargets,
    Nutrient,
    NutrientName,
)
from dietary_advisor.schemas.profile import (
    Allergen,
    Condition,
    DietPattern,
    Goal,
    Sex,
    UserProfile,
)

__all__ = [
    "Allergen",
    "Citation",
    "Condition",
    "DietPattern",
    "FoodItem",
    "Goal",
    "HardConstraint",
    "MacroTargets",
    "Meal",
    "MealKind",
    "MealPlan",
    "Nutrient",
    "NutrientName",
    "NutrientTotals",
    "Portion",
    "Recipe",
    "Sex",
    "SoftConstraint",
    "UserProfile",
    "ValidationReport",
    "Violation",
]
