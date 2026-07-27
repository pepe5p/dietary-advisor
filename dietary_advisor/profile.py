"""User profile schemas: the System Zarządzania Profilem layer.

These types are also surfaced to the LLM (via pydantic-ai) so the model can read
allergens, conditions and macro targets when generating a `MealPlan`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dietary_advisor.totaller.nutrition import MacroTargets


class ActivityLevel(str, Enum):
    """Self-reported activity level, surfaced to the LLM as context only.

    Not used for any deterministic calculation (target kcal/macros are
    supplied directly on `UserProfile.targets`); this is descriptive input
    for the agent's narrative reasoning.
    """

    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    ACTIVE = "active"
    VERY_ACTIVE = "very_active"


class UserProfile(BaseModel):
    """Full user profile - the deterministic source of truth for personalization"""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=64)
    age: int = Field(ge=1, le=120)
    sex: str = Field(min_length=1)
    height_cm: float = Field(gt=50, lt=260)
    weight_kg: float = Field(gt=2, lt=400)
    activity_level: ActivityLevel = ActivityLevel.MODERATE

    allergens: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    diet_pattern: str = "omnivore"
    disliked_foods: list[str] = Field(default_factory=list)
    preferred_foods: list[str] = Field(default_factory=list)

    # Target kcal/macros are supplied directly - calculating them from the
    # profile (e.g. via TDEE) is out of scope for this project.
    targets: MacroTargets

    notes: str | None = None

    @model_validator(mode="after")
    def _normalize(self) -> UserProfile:
        # Deduplicate while preserving order.
        self.disliked_foods = list(dict.fromkeys(s.lower().strip() for s in self.disliked_foods if s.strip()))
        self.preferred_foods = list(dict.fromkeys(s.lower().strip() for s in self.preferred_foods if s.strip()))
        self.allergens = list(dict.fromkeys(s.lower().strip() for s in self.allergens if s.strip()))
        self.conditions = list(dict.fromkeys(s.lower().strip() for s in self.conditions if s.strip()))
        # "none" is mutually exclusive with any other condition.
        if "none" in self.conditions and len(self.conditions) > 1:
            self.conditions = [c for c in self.conditions if c != "none"]
        return self
