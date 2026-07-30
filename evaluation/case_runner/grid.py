"""Hardcoded (model, variant, scenario) grid for case collection."""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.planning.pipeline import VariantConfig

BASELINE = VariantConfig(totaller_enabled=False, reflection_enabled=False)
FULL = VariantConfig()


@dataclass(frozen=True)
class RunSpec:
    llm_model: str
    variant: VariantConfig
    scenario_id: str


def planned_runs() -> list[RunSpec]:
    return [
        RunSpec(
            llm_model="gemini-3.5-flash-lite",
            variant=BASELINE,
            scenario_id="regular",
        ),
        RunSpec(
            llm_model="gemini-3.5-flash-lite",
            variant=FULL,
            scenario_id="regular",
        ),
        RunSpec(
            llm_model="gemini-3.6-flash",
            variant=BASELINE,
            scenario_id="regular",
        ),
    ]
