"""JSON persistence for successful case-runner results under ``output_dir/runs/``."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.totaller.nutrition import MacroTargets
from evaluation.case_runner.grid import RunSpec
from evaluation.persistence import artifact_path, is_present, load_json, save_json
from evaluation.validation.quantitative import NutrientErrors


class RunRecord(BaseModel):
    """One successful pipeline run, including judge-independent error metrics."""

    model_config = ConfigDict(extra="forbid")

    llm_model: str
    variant: str
    totaller_enabled: bool
    rag_enabled: bool
    reflection_enabled: bool
    scenario_id: str
    query: str
    agent_plan: AgentMealPlan
    targets: MacroTargets
    mae_pct: float
    mse_pct: float
    per_nutrient_pct: dict[str, float]
    iterations: int
    telemetry: dict[str, Any]
    elapsed_s: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def path_for(spec: RunSpec, *, output_dir: Path | None = None) -> Path:
    return artifact_path(spec, output_dir=output_dir, subdir="runs")


def is_done(spec: RunSpec, *, output_dir: Path | None = None) -> bool:
    return is_present(path_for(spec, output_dir=output_dir))


def save(spec: RunSpec, record: RunRecord, *, output_dir: Path | None = None) -> Path:
    return save_json(path_for(spec, output_dir=output_dir), record)


def load(spec: RunSpec, *, output_dir: Path | None = None) -> RunRecord:
    return load_json(path_for(spec, output_dir=output_dir), RunRecord)


def record_from_result(
    *,
    spec: RunSpec,
    query: str,
    agent_plan: AgentMealPlan,
    targets: MacroTargets,
    iterations: int,
    telemetry: dict[str, Any],
    elapsed_s: float,
    errors: NutrientErrors,
) -> RunRecord:
    v = spec.variant
    return RunRecord(
        llm_model=str(spec.llm),
        variant=v.label,
        totaller_enabled=v.totaller_enabled,
        rag_enabled=v.rag_enabled,
        reflection_enabled=v.reflection_enabled,
        scenario_id=spec.scenario_id,
        query=query,
        agent_plan=agent_plan,
        targets=targets,
        mae_pct=errors.mae,
        mse_pct=errors.mse,
        per_nutrient_pct=errors.per_nutrient,
        iterations=iterations,
        telemetry=telemetry,
        elapsed_s=elapsed_s,
    )
