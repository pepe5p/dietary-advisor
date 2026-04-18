"""Business logic on top of `ProfileStore` - notably constraint derivation.

`derive_hard_constraints` is the bridge from the user-facing profile to the
machine-readable rules consumed by the Validator (Pętla Walidacyjna). It
encodes the safety floors mentioned in the literature (e.g. WHO sodium cap,
ADA fiber floor for type-2 diabetes).
"""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.profile_manager.store import ProfileStore
from dietary_advisor.schemas.constraints import ConstraintSource, HardConstraint
from dietary_advisor.schemas.nutrition import NutrientName
from dietary_advisor.schemas.profile import Condition, DietPattern, UserProfile


# Clinical safety constants. These are conservative defaults sourced from the
# literature in the kwerenda; they are *not* clinical guidelines themselves
# and must be cross-validated by the RAG layer before patient-facing use.
_CONDITION_RULES: dict[Condition, list[HardConstraint]] = {
    Condition.HYPERTENSION: [
        HardConstraint.max_nutrient(
            NutrientName.SODIUM_MG,
            value=2000.0,  # WHO recommendation for adults with hypertension
            rationale="WHO recommends <2 g/day sodium for adults with hypertension.",
        ),
    ],
    Condition.TYPE_2_DIABETES: [
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
    Condition.DYSLIPIDEMIA: [
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
    Condition.CKD_STAGE_3: [
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
    Condition.CELIAC: [
        HardConstraint.allergen("gluten", source=ConstraintSource.SAFETY),
    ],
    Condition.LACTOSE_INTOLERANCE: [
        HardConstraint.allergen("milk", source=ConstraintSource.SAFETY),
    ],
}


@dataclass
class ProfileService:
    """Operations that combine storage with constraint derivation."""

    store: ProfileStore

    @classmethod
    def default(cls) -> ProfileService:
        return cls(store=ProfileStore())

    # --- Plain CRUD passthrough (kept here so callers depend on the service only).

    def upsert(self, profile: UserProfile) -> None:
        self.store.upsert(profile)

    def get(self, user_id: str) -> UserProfile | None:
        return self.store.get(user_id)

    def list_all(self) -> list[UserProfile]:
        return self.store.list_all()

    def delete(self, user_id: str) -> bool:
        return self.store.delete(user_id)

    # --- Constraint derivation.

    def derive_hard_constraints(self, profile: UserProfile) -> list[HardConstraint]:
        """Enumerate all hard constraints implied by the profile.

        The result combines:
          * one allergen_exclusion per declared allergen
          * one diet_pattern constraint when the user is not omnivore
          * one ingredient_exclusion per disliked food
          * any clinical-condition rules from `_CONDITION_RULES`
        """
        out: list[HardConstraint] = [
            HardConstraint.allergen(a.value) for a in profile.allergens
        ]
        if profile.diet_pattern != DietPattern.OMNIVORE:
            out.append(HardConstraint.diet(profile.diet_pattern.value))
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
