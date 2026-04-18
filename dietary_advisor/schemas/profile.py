"""User profile schemas: the System Zarządzania Profilem layer.

These types are also surfaced to the LLM (via pydantic-ai) so the model can read
allergens, conditions and goals when generating a `MealPlan`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Sex(str, Enum):
    """Biological sex, required by the Mifflin-St Jeor TDEE formula."""

    MALE = "male"
    FEMALE = "female"


class Allergen(str, Enum):
    """The 14 EU-regulated allergens (Annex II of EU Regulation 1169/2011)."""

    GLUTEN = "gluten"
    CRUSTACEANS = "crustaceans"
    EGGS = "eggs"
    FISH = "fish"
    PEANUTS = "peanuts"
    SOYBEANS = "soybeans"
    MILK = "milk"
    TREE_NUTS = "tree_nuts"
    CELERY = "celery"
    MUSTARD = "mustard"
    SESAME = "sesame"
    SULPHITES = "sulphites"
    LUPIN = "lupin"
    MOLLUSCS = "molluscs"


class Condition(str, Enum):
    """Clinical conditions relevant to dietetic guidance (Level 3 of the study)."""

    NONE = "none"
    TYPE_2_DIABETES = "type_2_diabetes"
    TYPE_1_DIABETES = "type_1_diabetes"
    HYPERTENSION = "hypertension"
    DYSLIPIDEMIA = "dyslipidemia"
    CKD_STAGE_3 = "ckd_stage_3"
    CELIAC = "celiac"
    LACTOSE_INTOLERANCE = "lactose_intolerance"
    IBS = "ibs"
    OBESITY = "obesity"


class DietPattern(str, Enum):
    """Diet pattern preferences / restrictions (Level 2 of the study)."""

    OMNIVORE = "omnivore"
    PESCATARIAN = "pescatarian"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    KETO = "keto"
    MEDITERRANEAN = "mediterranean"
    DASH = "dash"
    LOW_FODMAP = "low_fodmap"


class GoalKind(str, Enum):
    MAINTAIN = "maintain"
    LOSE = "lose"
    GAIN = "gain"


class Goal(BaseModel):
    """High-level dietary goal, used to derive caloric/macro targets."""

    kind: GoalKind = GoalKind.MAINTAIN
    target_kg: float | None = Field(
        default=None,
        description="Target body weight in kg, only meaningful for LOSE/GAIN.",
    )
    weekly_rate_kg: float = Field(
        default=0.5,
        ge=0.0,
        le=1.5,
        description="Desired weight change per week, in kg (capped at 1.5 for safety).",
    )


class UserProfile(BaseModel):
    """Full user profile - the deterministic source of truth for personalization.

    Per the thesis concept, this is the input on which all three complexity levels
    operate. Level 1 only requires demographics + activity; Level 2 adds allergens
    and a diet pattern; Level 3 adds clinical conditions.
    """

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    user_id: str = Field(min_length=1, max_length=64)
    name: str | None = None
    age: int = Field(ge=1, le=120)
    sex: Sex
    height_cm: float = Field(gt=50, lt=260)
    weight_kg: float = Field(gt=2, lt=400)
    activity_factor: float = Field(
        default=1.55,
        ge=1.2,
        le=2.4,
        description=("Mifflin-St Jeor PAL: 1.2 sedentary, 1.375 light, 1.55 moderate, 1.725 active, 1.9 very active."),
    )

    allergens: list[Allergen] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    diet_pattern: DietPattern = DietPattern.OMNIVORE
    disliked_foods: list[str] = Field(default_factory=list)
    preferred_foods: list[str] = Field(default_factory=list)
    goal: Goal = Field(default_factory=Goal)

    notes: str | None = None

    @model_validator(mode="after")
    def _normalize(self) -> UserProfile:
        # Deduplicate while preserving order.
        self.disliked_foods = list(dict.fromkeys(s.lower().strip() for s in self.disliked_foods if s.strip()))
        self.preferred_foods = list(dict.fromkeys(s.lower().strip() for s in self.preferred_foods if s.strip()))
        # NONE is mutually exclusive with any other condition.
        if Condition.NONE in self.conditions and len(self.conditions) > 1:
            self.conditions = [c for c in self.conditions if c != Condition.NONE]
        return self

    @property
    def complexity_level(self) -> int:
        """Map profile to the three thesis-defined complexity levels."""
        has_clinical = any(c != Condition.NONE for c in self.conditions)
        if has_clinical:
            return 3
        if self.allergens or self.diet_pattern != DietPattern.OMNIVORE:
            return 2
        return 1
