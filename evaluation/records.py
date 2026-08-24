"""Join a stored run with one judge's averaged verdict.

Error metrics live on the run because they are judge-independent; the reporting
layer reads both halves through one object.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evaluation.case_runner.grid import RunSpec
from evaluation.case_runner.store import load as load_run
from evaluation.case_runner.store import RunRecord
from evaluation.judges import Judge
from evaluation.scoring.store import is_scored, ScoreRecord
from evaluation.scoring.store import load as load_score
from evaluation.validation.qualitative import QualitativeResult


@dataclass(frozen=True)
class ScoredRun:
    run: RunRecord
    score: ScoreRecord

    @property
    def llm_model(self) -> str:
        return self.score.llm_model

    @property
    def variant(self) -> str:
        return self.score.variant

    @property
    def scenario_id(self) -> str:
        return self.score.scenario_id

    @property
    def iterations(self) -> int:
        return self.score.iterations

    @property
    def elapsed_s(self) -> float:
        return self.score.elapsed_s

    @property
    def qualitative(self) -> QualitativeResult:
        return self.score.qualitative

    @property
    def mae_pct(self) -> float:
        return self.run.mae_pct

    @property
    def mse_pct(self) -> float:
        return self.run.mse_pct

    @property
    def per_nutrient_pct(self) -> dict[str, float]:
        return self.run.per_nutrient_pct


def load_scored_run(spec: RunSpec, *, judge: Judge, output_dir: Path | None = None) -> ScoredRun:
    return ScoredRun(
        run=load_run(spec, output_dir=output_dir),
        score=load_score(spec, judge=judge, output_dir=output_dir),
    )


def scored_runs(
    specs: Sequence[RunSpec],
    *,
    judge: Judge,
    output_dir: Path | None = None,
) -> list[tuple[RunSpec, ScoredRun]]:
    return [
        (spec, load_scored_run(spec, judge=judge, output_dir=output_dir))
        for spec in specs
        if is_scored(spec, judge=judge, output_dir=output_dir)
    ]
