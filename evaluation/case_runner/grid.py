"""Hardcoded (model, variant, scenario) grid for case collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.scenarios import SCENARIOS

BASELINE = VariantConfig(totaller_enabled=False, reflection_enabled=False)
FULL_VARIANT = VariantConfig()
VARIANTS = [
    BASELINE,
    VariantConfig(totaller_enabled=True, reflection_enabled=False),
    VariantConfig(totaller_enabled=False, reflection_enabled=True),
    FULL_VARIANT,
]

ABLATION_MODELS = ["openrouter:openai/gpt-5.6-luna"]
MODEL_COMPARISON_MODELS = [
    "openrouter:google/gemini-3.5-flash-lite",
    "openrouter:google/gemini-3.6-flash",
    "openrouter:openai/gpt-5.6-luna",
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
    rep: int = 0

    @property
    def sanitized_llm_model(self) -> str:
        return self.llm_model.replace(":", "-").replace("/", "-")

    @property
    def spec_id(self) -> tuple[str, str, str, str]:
        return (self.sanitized_llm_model, self.variant.label, self.scenario_id, f"rep{self.rep}")

    @property
    def spec_key(self) -> str:
        return "__".join(self.spec_id)

    def __hash__(self) -> int:
        return hash(self.spec_id)


def _specs(
    llm_model: str,
    variant: VariantConfig,
    scenarios: list[str],
    reps: int,
) -> set[RunSpec]:
    return {
        RunSpec(llm_model=llm_model, variant=variant, scenario_id=scenario, rep=rep)
        for scenario in scenarios
        for rep in range(reps)
    }


def generate_minimal_scenarios_specs(llm_model: str, variant: VariantConfig, reps: int = 1) -> set[RunSpec]:
    return _specs(llm_model, variant, MINIMAL_SCENARIOS, reps)


def generate_all_scenarios_specs(llm_model: str, variant: VariantConfig, reps: int = 1) -> set[RunSpec]:
    return _specs(llm_model, variant, list(SCENARIOS), reps)


def create_experiment_1_specs() -> set[RunSpec]:
    """
    Experiment #1 is an ablation study that answers the question:
    How much does each component of the pipeline contribute to the overall performance?
    """
    result = set()

    for model in ABLATION_MODELS:
        for variant in VARIANTS:
            specs = generate_minimal_scenarios_specs(
                llm_model=model,
                variant=variant,
                reps=3,
            )
            result.update(specs)

    return result


def create_experiment_2_specs() -> set[RunSpec]:
    """
    Experiment #2 is measuring the performance with different models.
    """
    result = set()

    for model in MODEL_COMPARISON_MODELS:
        specs = generate_minimal_scenarios_specs(
            llm_model=model,
            variant=FULL_VARIANT,
            reps=3,
        )
        result.update(specs)

    return result


EXPERIMENTS: dict[str, Callable[[], set[RunSpec]]] = {
    "ablation": create_experiment_1_specs,
    "models": create_experiment_2_specs,
}


def experiment_runs(name: str) -> list[RunSpec]:
    return sorted(EXPERIMENTS[name](), key=lambda spec: spec.spec_key)


def planned_runs() -> list[RunSpec]:
    all_specs: set[RunSpec] = set()
    for factory in EXPERIMENTS.values():
        all_specs.update(factory())
    return sorted(all_specs, key=lambda spec: spec.spec_key)
