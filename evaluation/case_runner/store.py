"""JSON persistence for successful case-runner results under `outputs/`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.totaller.nutrition import MacroTargets
from evaluation.case_runner.grid import RunSpec

DEFAULT_OUTPUT_DIR = Path("outputs")


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


def _sanitize_model(model_id: str) -> str:
    return model_id.replace(":", "-").replace("/", "-")


def filename_for(llm_model: str, variant_label: str, scenario_id: str) -> str:
    return f"{_sanitize_model(llm_model)}__{variant_label}__{scenario_id}.json"


def path_for(spec: RunSpec, *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return output_dir / filename_for(spec.llm_model, spec.variant.label, spec.scenario_id)


def is_done(spec: RunSpec, *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> bool:
    return path_for(spec, output_dir=output_dir).is_file()


def save(record: RunRecord, *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename_for(record.llm_model, record.variant, record.scenario_id)
    dest.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return dest


def load(spec: RunSpec, *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> RunRecord:
    return RunRecord.model_validate_json(path_for(spec, output_dir=output_dir).read_text(encoding="utf-8"))


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
