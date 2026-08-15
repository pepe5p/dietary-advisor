"""Tests for score JSON store helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from dietary_advisor.config.llm import LlmSpec
from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.case_runner.grid import RunSpec
from evaluation.judges import all_judges, JUDGE_REPS
from evaluation.scoring.store import (
    has_rep,
    is_fully_scored,
    is_scored,
    load,
    load_rep,
    path_for,
    save,
    scored_reps,
    ScoreRecord,
)
from evaluation.validation.qualitative import CriterionScore, QualitativeResult

JUDGE_A, JUDGE_B = all_judges()[0], all_judges()[1]


def _record(*, soft: float = 0.9) -> ScoreRecord:
    return ScoreRecord(
        llm_model="test-model",
        variant="totaller+reflective-loop",
        scenario_id="regular",
        judge_model="test-judge",
        mae_pct=5.5,
        mse_pct=42.0,
        per_nutrient_pct={"energy_kcal": 3.0},
        qualitative=QualitativeResult(
            scores=[
                CriterionScore(
                    criterion_id="recipe-makes-sense",
                    score=soft,
                    reasoning="Looks fine.",
                ),
            ],
            aggregate=soft,
        ),
        iterations=1,
        elapsed_s=2.5,
    )


def test_path_for_uses_jrep_suffix(tmp_path: Path) -> None:
    spec = RunSpec(
        llm=LlmSpec(model="groq:llama-3.3-70b-versatile"),
        variant=VariantConfig(),
        scenario_id="regular",
    )
    path = path_for(spec, judge=JUDGE_A, rep=1, output_dir=tmp_path)
    assert path.parent == tmp_path / JUDGE_A.scores_subdir
    assert path.name.endswith("__jrep1.json")


def test_each_judge_gets_its_own_subdir(tmp_path: Path) -> None:
    spec = RunSpec(
        llm=LlmSpec(model="test-model"),
        variant=VariantConfig(),
        scenario_id="regular",
    )
    parents = {path_for(spec, judge=judge, rep=0, output_dir=tmp_path).parent for judge in all_judges()}
    assert len(parents) == len(all_judges())


def test_save_load_round_trip(tmp_path: Path) -> None:
    spec = RunSpec(llm=LlmSpec(model="test-model"), variant=VariantConfig(), scenario_id="regular")
    record = _record()
    dest = save(spec=spec, record=record, judge=JUDGE_A, rep=0, output_dir=tmp_path)
    assert dest.is_file()
    assert is_scored(spec=spec, judge=JUDGE_A, output_dir=tmp_path)
    assert not is_scored(spec=spec, judge=JUDGE_B, output_dir=tmp_path)
    assert not is_fully_scored(spec=spec, judge=JUDGE_A, output_dir=tmp_path)
    assert scored_reps(spec, judge=JUDGE_A, output_dir=tmp_path) == [0]

    loaded = load(spec=spec, judge=JUDGE_A, output_dir=tmp_path)
    assert loaded.mae_pct == record.mae_pct
    assert loaded.qualitative.aggregate == record.qualitative.aggregate


def test_load_averages_multiple_reps(tmp_path: Path) -> None:
    spec = RunSpec(llm=LlmSpec(model="test-model"), variant=VariantConfig(), scenario_id="regular")
    save(spec=spec, record=_record(soft=0.8), judge=JUDGE_A, rep=0, output_dir=tmp_path)
    save(spec=spec, record=_record(soft=1.0), judge=JUDGE_A, rep=1, output_dir=tmp_path)

    loaded = load(spec=spec, judge=JUDGE_A, output_dir=tmp_path)
    assert loaded.qualitative.aggregate == pytest.approx(0.9)
    assert has_rep(spec, judge=JUDGE_A, rep=0, output_dir=tmp_path)
    assert load_rep(spec, judge=JUDGE_A, rep=1, output_dir=tmp_path).qualitative.aggregate == pytest.approx(1.0)


def test_is_fully_scored_when_all_reps_present(tmp_path: Path) -> None:
    spec = RunSpec(llm=LlmSpec(model="test-model"), variant=VariantConfig(), scenario_id="regular")
    for rep in range(JUDGE_REPS):
        save(spec=spec, record=_record(), judge=JUDGE_A, rep=rep, output_dir=tmp_path)
    assert is_fully_scored(spec=spec, judge=JUDGE_A, output_dir=tmp_path)
    assert scored_reps(spec, judge=JUDGE_A, output_dir=tmp_path) == list(range(JUDGE_REPS))
