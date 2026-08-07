"""Tests for evaluation grid RunSpec generation."""

from __future__ import annotations

from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.case_runner.grid import generate_minimal_scenarios_specs, RunSpec


def test_reps_expand_into_distinct_specs() -> None:
    variant = VariantConfig()
    specs = generate_minimal_scenarios_specs(
        llm_model="test-model",
        variant=variant,
        reps=3,
    )
    assert len(specs) == 15
    regular_keys = {spec.spec_key for spec in specs if spec.scenario_id == "regular"}
    assert regular_keys == {
        "test-model__totaller+reflective-loop__regular__rep0",
        "test-model__totaller+reflective-loop__regular__rep1",
        "test-model__totaller+reflective-loop__regular__rep2",
    }


def test_reps_differ_only_by_rep_index() -> None:
    variant = VariantConfig()
    rep0 = RunSpec(llm_model="test-model", variant=variant, scenario_id="regular", rep=0)
    rep1 = RunSpec(llm_model="test-model", variant=variant, scenario_id="regular", rep=1)
    assert rep0 != rep1
    assert len({rep0, rep1}) == 2
