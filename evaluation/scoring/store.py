"""JSON persistence for scored case-run results under ``output_dir/scores/``."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evaluation.case_runner.grid import RunSpec
from evaluation.persistence import artifact_path, is_present, load_json, resolve_output_dir, save_json
from evaluation.validation.qualitative import QualitativeResult


class ScoreRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_model: str
    variant: str
    scenario_id: str
    judge_model: str
    mae_pct: float
    mse_pct: float
    per_nutrient_pct: dict[str, float]
    qualitative: QualitativeResult
    iterations: int
    elapsed_s: float
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def scores_dir(*, output_dir: Path | None = None) -> Path:
    return resolve_output_dir(output_dir) / "scores"


def path_for(spec: RunSpec, *, output_dir: Path | None = None) -> Path:
    return artifact_path(spec, output_dir=output_dir, subdir="scores")


def is_scored(spec: RunSpec, *, output_dir: Path | None = None) -> bool:
    return is_present(path_for(spec, output_dir=output_dir))


def save(spec: RunSpec, record: ScoreRecord, *, output_dir: Path | None = None) -> Path:
    return save_json(path_for(spec, output_dir=output_dir), record)


def load(spec: RunSpec, *, output_dir: Path | None = None) -> ScoreRecord:
    return load_json(path_for(spec, output_dir=output_dir), ScoreRecord)
