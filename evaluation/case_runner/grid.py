"""Hardcoded (model, variant, scenario) grid for case collection."""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.scenarios import SCENARIOS

BASELINE = VariantConfig(totaller_enabled=False, reflection_enabled=False)
FULL = VariantConfig()
VARIANTS = [
    BASELINE,
    VariantConfig(totaller_enabled=True, reflection_enabled=False),
    VariantConfig(totaller_enabled=False, reflection_enabled=True),
    FULL,
]


MINIMAL_SCENARIOS = [
    "regular",
    "lactose-intolerant-athlete-wants-cheesecake",
    "vegetarian-allergic",
    "diabetes-hypertension",
    "dyslipidemia-obesity",
]


@dataclass(frozen=True)
class RunSpec:
    llm_model: str
    variant: VariantConfig
    scenario_id: str

    @property
    def sanitized_llm_model(self) -> str:
        return self.llm_model.replace(":", "-").replace("/", "-")

    @property
    def spec_id(self) -> tuple[str, str, str]:
        return (self.sanitized_llm_model, self.variant.label, self.scenario_id)

    @property
    def spec_key(self) -> str:
        return "__".join(self.spec_id)

    def __hash__(self) -> int:
        return hash(self.spec_id)


def generate_minimal_scenarios_specs(llm_model: str, variant: VariantConfig) -> set[RunSpec]:
    return {
        RunSpec(
            llm_model=llm_model,
            variant=variant,
            scenario_id=scenario,
        )
        for scenario in MINIMAL_SCENARIOS
    }


def generate_all_scenarios_specs(llm_model: str, variant: VariantConfig) -> set[RunSpec]:
    return {
        RunSpec(
            llm_model=llm_model,
            variant=variant,
            scenario_id=scenario,
        )
        for scenario in SCENARIOS
    }


def create_experiment_1_specs() -> set[RunSpec]:
    """
    Experiment #1 is an ablation study that answers the question:
    How much does each component of the pipeline contribute to the overall performance?
    """
    result = set()

    for model in ["openrouter:openai/gpt-5.6-luna"]:
        for variant in VARIANTS:
            specs = generate_minimal_scenarios_specs(
                llm_model=model,
                variant=variant,
            )
            result.update(specs)

    return result


def planned_runs() -> list[RunSpec]:
    all_specs = set()
    all_specs.update(create_experiment_1_specs())
    return list(all_specs)
