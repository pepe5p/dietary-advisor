"""Code-based realisation of `HardConstraint`s.

Each rule type knows how to inspect a `MealPlan` (and its `NutrientTotals`)
and emit zero or more `Violation`s. This is intentionally *not* an LLM:
relying on a probabilistic checker would defeat the whole purpose of the
neuro-symbolic architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from dietary_advisor.schemas.meal_plan import MealPlan, NutrientTotals
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName
from evaluation.constraints import HardConstraint, Violation


def food_contains_allergen(food: FoodItem, allergen: str) -> bool:
    """Cheap allergen check based on canonical `contains:<allergen>` tags."""
    target = f"contains:{allergen.lower()}"
    return any(t.lower() == target for t in food.tags)


class HardRule(ABC):
    """Base class for executable hard rules."""

    constraint: HardConstraint

    @abstractmethod
    def check(self, plan: MealPlan, totals: NutrientTotals) -> list[Violation]: ...


@dataclass
class AllergenExclusionRule(HardRule):
    constraint: HardConstraint

    def check(self, plan: MealPlan, totals: NutrientTotals) -> list[Violation]:  # noqa: ARG002
        out: list[Violation] = []
        target = self.constraint.target.lower()
        for meal in plan.meals:
            for portion in meal.recipe.portions:
                if food_contains_allergen(portion.food, target):
                    out.append(
                        Violation(
                            constraint=self.constraint,
                            detail=(
                                f"{portion.food.name!r} in {meal.kind.value} contains the excluded allergen {target!r}."
                            ),
                            offending_item=portion.food.name,
                        ),
                    )
        return out


@dataclass
class IngredientExclusionRule(HardRule):
    constraint: HardConstraint

    def check(self, plan: MealPlan, totals: NutrientTotals) -> list[Violation]:  # noqa: ARG002
        out: list[Violation] = []
        needle = self.constraint.target.lower()
        for meal in plan.meals:
            for portion in meal.recipe.portions:
                if needle in portion.food.name.lower():
                    out.append(
                        Violation(
                            constraint=self.constraint,
                            detail=f"{portion.food.name!r} matches excluded ingredient {needle!r}.",
                            offending_item=portion.food.name,
                        ),
                    )
        return out


@dataclass
class DietPatternRule(HardRule):
    """Every food must carry the diet-pattern tag (e.g. 'vegan')."""

    constraint: HardConstraint

    def check(self, plan: MealPlan, totals: NutrientTotals) -> list[Violation]:  # noqa: ARG002
        out: list[Violation] = []
        target = self.constraint.target.lower()
        for meal in plan.meals:
            for portion in meal.recipe.portions:
                tags = {t.lower() for t in portion.food.tags}
                if target not in tags:
                    out.append(
                        Violation(
                            constraint=self.constraint,
                            detail=(
                                f"{portion.food.name!r} is not tagged {target!r} (tags: {sorted(tags) or 'none'})."
                            ),
                            offending_item=portion.food.name,
                        ),
                    )
        return out


@dataclass
class MaxNutrientRule(HardRule):
    constraint: HardConstraint

    def check(self, plan: MealPlan, totals: NutrientTotals) -> list[Violation]:  # noqa: ARG002
        if self.constraint.value is None:
            return []
        try:
            nutrient = NutrientName(self.constraint.target)
        except ValueError:
            return []
        actual = totals.get(nutrient)
        if actual > self.constraint.value:
            return [
                Violation(
                    constraint=self.constraint,
                    detail=(f"Total {nutrient.value} = {actual:.2f} exceeds maximum {self.constraint.value:.2f}."),
                    offending_value=actual,
                ),
            ]
        return []


@dataclass
class MinNutrientRule(HardRule):
    constraint: HardConstraint

    def check(self, plan: MealPlan, totals: NutrientTotals) -> list[Violation]:  # noqa: ARG002
        if self.constraint.value is None:
            return []
        try:
            nutrient = NutrientName(self.constraint.target)
        except ValueError:
            return []
        actual = totals.get(nutrient)
        if actual < self.constraint.value:
            return [
                Violation(
                    constraint=self.constraint,
                    detail=(f"Total {nutrient.value} = {actual:.2f} is below minimum {self.constraint.value:.2f}."),
                    offending_value=actual,
                ),
            ]
        return []


@dataclass
class MealCountRule(HardRule):
    """The plan must contain exactly `value` meals (the requested number)."""

    constraint: HardConstraint

    def check(self, plan: MealPlan, totals: NutrientTotals) -> list[Violation]:  # noqa: ARG002
        if self.constraint.value is None:
            return []
        required = int(self.constraint.value)
        actual = len(plan.meals)
        if actual != required:
            return [
                Violation(
                    constraint=self.constraint,
                    detail=f"Plan has {actual} meal(s) but exactly {required} were requested.",
                    offending_value=float(actual),
                ),
            ]
        return []


def rule_from_constraint(constraint: HardConstraint) -> HardRule:
    """Factory dispatch from constraint kind to rule implementation."""
    match constraint.kind:
        case "allergen_exclusion":
            return AllergenExclusionRule(constraint)
        case "ingredient_exclusion":
            return IngredientExclusionRule(constraint)
        case "diet_pattern":
            return DietPatternRule(constraint)
        case "max_nutrient":
            return MaxNutrientRule(constraint)
        case "min_nutrient":
            return MinNutrientRule(constraint)
        case "meal_count":
            return MealCountRule(constraint)
    raise ValueError(f"Unknown constraint kind: {constraint.kind}")


def rules_from_constraints(constraints: list[HardConstraint]) -> list[HardRule]:
    return [rule_from_constraint(c) for c in constraints]
