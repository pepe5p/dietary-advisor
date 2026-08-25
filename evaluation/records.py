"""Join a stored run with one judge's averaged verdict.

Error metrics live on the run because they are judge-independent; the reporting
layer reads both halves through one object.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dietary_advisor.agents.nutrition.agent import total_meal_plan
from evaluation.case_runner.grid import RunSpec
from evaluation.case_runner.store import load as load_run
from evaluation.case_runner.store import RunRecord
from evaluation.judges import Judge
from evaluation.scoring.store import is_scored, ScoreRecord
from evaluation.scoring.store import load as load_score
from evaluation.validation.qualitative import QualitativeResult

_TOTALLER_TOOL = total_meal_plan.__name__


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
    def totaller_enabled(self) -> bool:
        return self.run.totaller_enabled

    @property
    def totaller_calls(self) -> int:
        tool_calls = self.run.telemetry.get("tool_calls", {})
        return int(tool_calls.get(_TOTALLER_TOOL, 0))

    def _telemetry_int(self, key: str) -> int:
        value = self.run.telemetry.get(key)
        return 0 if value is None else int(value)

    @property
    def input_tokens(self) -> int:
        return self._telemetry_int("input_tokens")

    @property
    def output_tokens(self) -> int:
        return self._telemetry_int("output_tokens")

    @property
    def cache_read_tokens(self) -> int:
        return self._telemetry_int("cache_read_tokens")

    @property
    def reasoning_tokens(self) -> int:
        return self._telemetry_int("reasoning_tokens")

    # Subtractions are clamped: cache-read can exceed input, and reasoning
    # can be missing or exceed output.
    @property
    def fresh_input_tokens(self) -> int:
        return max(self.input_tokens - self.cache_read_tokens, 0)

    @property
    def visible_output_tokens(self) -> int:
        return max(self.output_tokens - self.reasoning_tokens, 0)

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
