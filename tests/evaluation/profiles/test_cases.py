"""Tests for frozen evaluation cases."""

from __future__ import annotations

from evaluation.profiles.cases import all_cases, EVAL_CASES, get_case
from evaluation.profiles.derive import derive_hard_constraints


def test_eval_cases_count() -> None:
    assert len(EVAL_CASES) == 15
    assert len(all_cases()) == 15


def test_get_case_returns_same_profile() -> None:
    c = get_case("L1_01")
    assert c.case_id == "L1_01"
    assert c.profile.user_id == "L1_01"


def test_hard_constraints_match_derivation() -> None:
    """Guard: the frozen `hard_constraints` in cases.py must match what derives from the profile."""
    for eval_profile in all_cases():
        derived = derive_hard_constraints(eval_profile.profile)
        assert len(eval_profile.hard_constraints) == len(derived)
        derived_keys = {(c.kind, c.target) for c in derived}
        for c in eval_profile.hard_constraints:
            assert (c.kind, c.target) in derived_keys
