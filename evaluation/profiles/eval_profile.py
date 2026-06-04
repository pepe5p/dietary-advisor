"""Frozen evaluation profile: user data plus ground-truth rules and targets."""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.schemas.constraints import HardConstraint
from dietary_advisor.schemas.meal_plan import MealKind
from dietary_advisor.schemas.nutrition import MacroTargets
from dietary_advisor.schemas.profile import UserProfile


def case_complexity(case_id: str) -> int:
    """Derive ablation complexity level from case id prefix (L1_01 → 1)."""
    return int(case_id[1])


@dataclass(frozen=True)
class EvalProfile:
    """Test profile with frozen hard constraints, macro targets, and default query."""

    case_id: str
    profile: UserProfile
    hard_constraints: tuple[HardConstraint, ...]
    macro_targets: MacroTargets
    default_query: str
    forbidden_meal_kinds: frozenset[MealKind] = frozenset()
