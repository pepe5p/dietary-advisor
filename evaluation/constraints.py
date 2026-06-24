"""Constraint schemas: the ground-truth rules the evaluation Validator enforces.

Hard constraints (HSR) are non-negotiable - violating them is a critical failure
(e.g. exposing an allergic patient to peanuts). Soft constraints (SSR) shape
ranking and may be partially relaxed.

These are evaluation-only: the production pipeline never sees them (the agent
must infer restrictions from the profile itself); they exist solely as the
frozen ground truth attached to each `EvalProfile` for scoring CSR.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.schemas.nutrition import NutrientAmountMap, NutrientName


class ConstraintSource(str, Enum):
    """Where the constraint originated, for explainability."""

    PROFILE = "profile"  # User-declared (allergen, diet pattern, ...)
    CLINICAL_GUIDELINE = "clinical_guideline"  # Pulled from RAG (WHO/NICE/ADA)
    GOAL = "goal"  # Derived from caloric/macro targets
    SAFETY = "safety"  # Hardcoded safety rail


class HardConstraint(BaseModel):
    """A non-negotiable rule. Violation => HSR failure."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "allergen_exclusion",
        "diet_pattern",
        "max_nutrient",
        "min_nutrient",
        "ingredient_exclusion",
        "meal_count",
    ]
    # `target` semantics depend on `kind`:
    #   allergen_exclusion: the allergen string (e.g. "peanuts")
    #   diet_pattern:       the diet pattern string (e.g. "vegetarian")
    #   max_nutrient/min_nutrient: the NutrientName value (e.g. "sodium_mg")
    #   ingredient_exclusion: the ingredient name (lowercased)
    #   meal_count:         the literal "meal_count" (count carried in `value`)
    target: str = Field(min_length=1)
    # Numeric threshold for max_nutrient / min_nutrient, or the exact required
    # number of meals for meal_count; ignored otherwise.
    value: float | None = None
    source: ConstraintSource = ConstraintSource.PROFILE
    rationale: str | None = None

    @classmethod
    def allergen(cls, allergen: str, source: ConstraintSource = ConstraintSource.PROFILE) -> HardConstraint:
        return cls(kind="allergen_exclusion", target=allergen.lower(), source=source)

    @classmethod
    def diet(cls, pattern: str, source: ConstraintSource = ConstraintSource.PROFILE) -> HardConstraint:
        return cls(kind="diet_pattern", target=pattern.lower(), source=source)

    @classmethod
    def max_nutrient(
        cls,
        nutrient: NutrientName,
        value: float,
        source: ConstraintSource = ConstraintSource.CLINICAL_GUIDELINE,
        rationale: str | None = None,
    ) -> HardConstraint:
        return cls(kind="max_nutrient", target=nutrient.value, value=value, source=source, rationale=rationale)

    @classmethod
    def min_nutrient(
        cls,
        nutrient: NutrientName,
        value: float,
        source: ConstraintSource = ConstraintSource.CLINICAL_GUIDELINE,
        rationale: str | None = None,
    ) -> HardConstraint:
        return cls(kind="min_nutrient", target=nutrient.value, value=value, source=source, rationale=rationale)

    @classmethod
    def meal_count(
        cls,
        count: int,
        source: ConstraintSource = ConstraintSource.PROFILE,
        rationale: str | None = None,
    ) -> HardConstraint:
        return cls(kind="meal_count", target="meal_count", value=float(count), source=source, rationale=rationale)


class SoftConstraint(BaseModel):
    """A preference that nudges ranking but doesn't fail HSR if violated."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["prefer_ingredient", "avoid_ingredient", "near_target_nutrient"]
    target: str
    value: float | None = None
    weight: float = Field(default=1.0, ge=0.0)


class Violation(BaseModel):
    """A concrete violation produced by the Validator."""

    model_config = ConfigDict(extra="forbid")

    constraint: HardConstraint
    detail: str
    # If applicable: the offending nutrient amount or food name.
    offending_value: float | None = None
    offending_item: str | None = None


class ValidationReport(BaseModel):
    """Full result returned by `validate_meal_plan`."""

    model_config = ConfigDict(extra="forbid")

    hard_satisfied: bool
    violations: list[Violation] = Field(default_factory=list)
    # Per-nutrient totals, used by the evaluation harness for MAE/MSE too.
    totals: NutrientAmountMap = Field(default_factory=dict)
    # CSR is computed by `evaluation/validation/structural.py`.
    csr: float | None = None
    hsr: float | None = None
    ssr: float | None = None

    def summary(self) -> str:
        if self.hard_satisfied:
            return "OK: all hard constraints satisfied."
        bullets = "\n".join(f"  - {v.detail}" for v in self.violations)
        return f"FAIL: {len(self.violations)} hard constraint violation(s):\n{bullets}"
