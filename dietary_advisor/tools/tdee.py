"""Mifflin-St Jeor BMR + TDEE + macro target derivation.

These are *deterministic* formulas; pulling them out of the LLM is the whole
point of the Totaller (see thesis sec. "Moduł Obliczeniowy"). The functions
here are pure and trivially testable.

References:
    Mifflin MD et al. (1990) "A new predictive equation for resting energy
    expenditure in healthy individuals." Am J Clin Nutr 51:241-247.
"""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.schemas.nutrition import MacroTargets
from dietary_advisor.schemas.profile import GoalKind, Sex, UserProfile

# Calories per gram of macronutrient (Atwater factors).
KCAL_PER_G_PROTEIN = 4.0
KCAL_PER_G_CARBS = 4.0
KCAL_PER_G_FAT = 9.0


@dataclass(frozen=True)
class EnergyEstimate:
    """Bundle of derived energy figures."""

    bmr_kcal: float
    tdee_kcal: float
    target_kcal: float


def mifflin_st_jeor_bmr(*, sex: Sex, weight_kg: float, height_cm: float, age: int) -> float:
    """Resting BMR per Mifflin-St Jeor (kcal/day)."""
    base = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age
    if sex == Sex.MALE:
        return base + 5.0
    return base - 161.0


def estimate_energy(profile: UserProfile) -> EnergyEstimate:
    """Compute BMR, TDEE and a goal-adjusted target kcal for the profile.

    Goal adjustment uses 7,700 kcal per kg of body fat as the canonical
    energy density of adipose tissue, capped at +/- 25% of TDEE for safety.
    """
    bmr = mifflin_st_jeor_bmr(
        sex=profile.sex,
        weight_kg=profile.weight_kg,
        height_cm=profile.height_cm,
        age=profile.age,
    )
    tdee = bmr * profile.activity_factor

    weekly_delta_kcal = 7700.0 * profile.goal.weekly_rate_kg
    daily_delta = weekly_delta_kcal / 7.0
    if profile.goal.kind == GoalKind.LOSE:
        adj = -daily_delta
    elif profile.goal.kind == GoalKind.GAIN:
        adj = +daily_delta
    else:
        adj = 0.0

    cap = 0.25 * tdee
    adj = max(-cap, min(cap, adj))
    target = max(1200.0, tdee + adj)  # Soft floor for safety.
    return EnergyEstimate(bmr_kcal=bmr, tdee_kcal=tdee, target_kcal=target)


def derive_macro_targets(profile: UserProfile) -> MacroTargets:
    """Default macro split: 1.6 g/kg protein, 25% kcal fat, remainder carbs.

    These splits are conservative defaults aligned with WHO/USDA DGA ranges
    and are *not* clinical advice - they're the symbolic anchor against which
    we compute MAE in the evaluation harness.
    """
    energy = estimate_energy(profile)
    target_kcal = energy.target_kcal

    protein_g = round(1.6 * profile.weight_kg, 1)
    protein_kcal = protein_g * KCAL_PER_G_PROTEIN

    fat_kcal = 0.25 * target_kcal
    fat_g = round(fat_kcal / KCAL_PER_G_FAT, 1)

    carbs_kcal = max(0.0, target_kcal - protein_kcal - fat_kcal)
    carbs_g = round(carbs_kcal / KCAL_PER_G_CARBS, 1)

    # WHO recommends >=25 g/day fiber for adults.
    fiber_g = 25.0 if profile.age >= 18 else 14.0 * (target_kcal / 1000.0)

    return MacroTargets(
        energy_kcal=round(target_kcal, 1),
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        fiber_g=round(fiber_g, 1),
    )
