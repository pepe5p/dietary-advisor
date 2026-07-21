"""Hardcoded (model, variant, scenario) grid for case collection."""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.ablation import leave_one_out_variants
from evaluation.scenarios import SCENARIOS

# Extend this list to collect runs against additional pydantic-ai model ids.
MODELS: list[str] = ["gemini-3.1-flash-lite"]


@dataclass(frozen=True)
class RunSpec:
    llm_model: str
    variant: VariantConfig
    scenario_id: str


def planned_runs() -> list[RunSpec]:
    """Cartesian product of MODELS x leave-one-out variants x SCENARIOS."""
    return [
        RunSpec(llm_model=model, variant=variant, scenario_id=scenario.case_id)
        for model in MODELS
        for variant in leave_one_out_variants()
        for scenario in SCENARIOS
    ]
