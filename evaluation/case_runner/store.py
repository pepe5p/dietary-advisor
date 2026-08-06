"""JSON persistence for successful case-runner results under the configured output dir."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.totaller.nutrition import MacroTargets
from evaluation.case_runner.grid import RunSpec
from evaluation.persistence import artifact_path, is_present, load_json, save_json


class RunRecord(BaseModel):
    """Everything the later scoring stage needs from one successful pipeline run."""

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
    iterations: int
    telemetry: dict[str, Any]
    elapsed_s: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def path_for(spec: RunSpec, *, output_dir: Path | None = None) -> Path:
    return artifact_path(spec, output_dir=output_dir)


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
) -> RunRecord:
    v = spec.variant
    return RunRecord(
        llm_model=spec.llm_model,
        variant=v.label,
        totaller_enabled=v.totaller_enabled,
        rag_enabled=v.rag_enabled,
        reflection_enabled=v.reflection_enabled,
        scenario_id=spec.scenario_id,
        query=query,
        agent_plan=agent_plan,
        targets=targets,
        iterations=iterations,
        telemetry=telemetry,
        elapsed_s=elapsed_s,
    )
