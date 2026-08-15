"""JSON persistence for scored case-run results, one directory per judge."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evaluation.case_runner.grid import RunSpec
from evaluation.judges import Judge, JUDGE_REPS
from evaluation.persistence import is_present, load_json, resolve_output_dir, save_json
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


def scores_dir(*, judge: Judge, output_dir: Path | None = None) -> Path:
    return resolve_output_dir(output_dir) / judge.scores_subdir


def path_for(spec: RunSpec, *, judge: Judge, rep: int, output_dir: Path | None = None) -> Path:
    return scores_dir(judge=judge, output_dir=output_dir) / f"{spec.spec_key}__jrep{rep}.json"


def has_rep(spec: RunSpec, *, judge: Judge, rep: int, output_dir: Path | None = None) -> bool:
    return is_present(path_for(spec, judge=judge, rep=rep, output_dir=output_dir))


def scored_reps(spec: RunSpec, *, judge: Judge, output_dir: Path | None = None) -> list[int]:
    return [rep for rep in range(JUDGE_REPS) if has_rep(spec, judge=judge, rep=rep, output_dir=output_dir)]


def is_scored(spec: RunSpec, *, judge: Judge, output_dir: Path | None = None) -> bool:
    return bool(scored_reps(spec, judge=judge, output_dir=output_dir))


def is_fully_scored(spec: RunSpec, *, judge: Judge, output_dir: Path | None = None) -> bool:
    return len(scored_reps(spec, judge=judge, output_dir=output_dir)) == JUDGE_REPS


def save(
    spec: RunSpec,
    record: ScoreRecord,
    *,
    judge: Judge,
    rep: int,
    output_dir: Path | None = None,
) -> Path:
    return save_json(path_for(spec, judge=judge, rep=rep, output_dir=output_dir), record)


def load_rep(spec: RunSpec, *, judge: Judge, rep: int, output_dir: Path | None = None) -> ScoreRecord:
    return load_json(path_for(spec, judge=judge, rep=rep, output_dir=output_dir), ScoreRecord)


def load(spec: RunSpec, *, judge: Judge, output_dir: Path | None = None) -> ScoreRecord:
    from evaluation.scoring.average import average_score_records

    reps = scored_reps(spec, judge=judge, output_dir=output_dir)
    if not reps:
        raise FileNotFoundError(f"No score reps for {spec.spec_key} ({judge.key})")
    records = [load_rep(spec, judge=judge, rep=rep, output_dir=output_dir) for rep in reps]
    return average_score_records(records)
