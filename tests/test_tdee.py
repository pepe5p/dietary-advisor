"""Mifflin-St Jeor + TDEE + macro target derivation."""

from __future__ import annotations

import pytest

from dietary_advisor.schemas.profile import GoalKind, Sex, UserProfile
from dietary_advisor.tools.tdee import (
    derive_macro_targets,
    estimate_energy,
    mifflin_st_jeor_bmr,
)


def test_mifflin_st_jeor_male() -> None:
    bmr = mifflin_st_jeor_bmr(sex=Sex.MALE, weight_kg=70, height_cm=175, age=30)
    # 10*70 + 6.25*175 - 5*30 + 5 = 700 + 1093.75 - 150 + 5 = 1648.75
    assert bmr == pytest.approx(1648.75)


def test_mifflin_st_jeor_female() -> None:
    bmr = mifflin_st_jeor_bmr(sex=Sex.FEMALE, weight_kg=60, height_cm=165, age=28)
    # 10*60 + 6.25*165 - 5*28 - 161 = 600 + 1031.25 - 140 - 161 = 1330.25
    assert bmr == pytest.approx(1330.25)


def test_estimate_energy_includes_activity_factor(healthy_profile: UserProfile) -> None:
    e = estimate_energy(healthy_profile)
    assert e.tdee_kcal == pytest.approx(e.bmr_kcal * healthy_profile.activity_factor)


def test_lose_goal_decreases_target_kcal() -> None:
    base = UserProfile(user_id="b", age=40, sex=Sex.MALE, height_cm=175, weight_kg=90)
    losing = base.model_copy(update={"goal": base.goal.model_copy(update={"kind": GoalKind.LOSE})})
    e_b = estimate_energy(base).target_kcal
    e_l = estimate_energy(losing).target_kcal
    assert e_l < e_b


def test_target_kcal_is_capped_safely() -> None:
    # Aggressive 1.5 kg/week on a low-TDEE profile should be capped at -25%.
    p = UserProfile(
        user_id="x",
        age=70,
        sex=Sex.FEMALE,
        height_cm=160,
        weight_kg=55,
        activity_factor=1.2,
    )
    p = p.model_copy(update={"goal": p.goal.model_copy(update={"kind": GoalKind.LOSE, "weekly_rate_kg": 1.5})})
    e = estimate_energy(p)
    assert e.target_kcal >= 0.75 * e.tdee_kcal - 1e-6


def test_macro_targets_sum_to_target_kcal() -> None:
    p = UserProfile(user_id="x", age=30, sex=Sex.MALE, height_cm=180, weight_kg=80)
    targets = derive_macro_targets(p)
    summed = targets.protein_g * 4 + targets.carbs_g * 4 + targets.fat_g * 9
    # Allow small rounding slack from the per-macro round(1).
    assert summed == pytest.approx(targets.energy_kcal, rel=0.02, abs=15)
