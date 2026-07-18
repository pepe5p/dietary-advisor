"""Frozen evaluation profile: user data plus ground-truth hard constraints."""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.schemas.profile import UserProfile
from evaluation.constraints import HardConstraint


@dataclass(frozen=True)
class EvalProfile:
    case_id: str
    profile: UserProfile
    hard_constraints: tuple[HardConstraint, ...]
