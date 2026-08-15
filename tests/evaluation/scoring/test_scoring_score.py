"""Tests for score_runs skip-existing, force, and failure behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from dietary_advisor.config.llm import LlmSpec
from dietary_advisor.food_db import FoodDb
from dietary_advisor.planning.pipeline import VariantConfig
from dietary_advisor.totaller.nutrition import MacroTargets
from evaluation.case_runner.grid import RunSpec
from evaluation.case_runner.store import RunRecord
from evaluation.case_runner.store import save as save_run
from evaluation.judges import all_judges
from evaluation.scoring.score import score_runs
from evaluation.scoring.store import is_scored, load
from evaluation.validation.qualitative import CriterionScore, QualitativeResult
from tests.evaluation.conftest import agent_plan_single

JUDGE_A, JUDGE_B = all_judges()[0], all_judges()[1]


def _run_spec(*, scenario_id: str = "regular") -> RunSpec:
    return RunSpec(
        llm=LlmSpec(model="test-model"),
        variant=VariantConfig(totaller_enabled=False, rag_enabled=False, reflection_enabled=False),
        scenario_id=scenario_id,
    )


def _seed_run(tmp_path: Path, spec: RunSpec, *, any_code: str) -> None:
    variant = spec.variant
    save_run(
        spec=spec,
        record=RunRecord(
            llm_model=str(spec.llm),
            variant=variant.label,
            totaller_enabled=variant.totaller_enabled,
            rag_enabled=variant.rag_enabled,
            reflection_enabled=variant.reflection_enabled,
            scenario_id=spec.scenario_id,
            query="",
            agent_plan=agent_plan_single(any_code, grams=200.0, user_id="regular"),
            targets=MacroTargets(energy_kcal=2000, protein_g=100, carbs_g=200, fat_g=70),
            iterations=1,
            telemetry={"requests": 1},
            elapsed_s=1.0,
        ),
        output_dir=tmp_path,
    )


def _qualitative(aggregate: float) -> QualitativeResult:
    return QualitativeResult(
        scores=[
            CriterionScore(
                criterion_id="recipe-makes-sense",
                score=aggregate,
                reasoning="Coherent.",
            ),
        ],
        aggregate=aggregate,
    )


def _patch_judges(monkeypatch: pytest.MonkeyPatch, *, j1: float = 0.9, j2: float = 0.5) -> None:
    by_model = {JUDGE_A.model_id: j1, JUDGE_B.model_id: j2}

    def fake_resolve(model_id: str) -> TestModel:
        return TestModel(custom_output_args=_qualitative(by_model[model_id]).model_dump())

    monkeypatch.setattr("evaluation.judges.resolve_judge_model", fake_resolve)


@pytest.mark.asyncio()
async def test_score_runs_writes_and_skips_existing(
    tmp_path: Path,
    food_db: FoodDb,
    any_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_judges(monkeypatch)
    spec = _run_spec()
    _seed_run(tmp_path, spec, any_code=any_code)

    first = await score_runs([spec], judges=all_judges(), output_dir=tmp_path, lookup=food_db)
    assert first[JUDGE_A.key].succeeded == 1
    assert first[JUDGE_B.key].succeeded == 1
    assert is_scored(spec, judge=JUDGE_A, output_dir=tmp_path)
    assert is_scored(spec, judge=JUDGE_B, output_dir=tmp_path)

    second = await score_runs([spec], judges=all_judges(), output_dir=tmp_path, lookup=food_db)
    assert second[JUDGE_A.key].already_scored == 1
    assert second[JUDGE_B.key].already_scored == 1
    assert second[JUDGE_A.key].attempted == 0


@pytest.mark.asyncio()
async def test_score_runs_force_rescores(
    tmp_path: Path,
    food_db: FoodDb,
    any_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_judges(monkeypatch, j1=0.9, j2=0.5)
    spec = _run_spec()
    _seed_run(tmp_path, spec, any_code=any_code)

    await score_runs([spec], judges=(JUDGE_A,), output_dir=tmp_path, lookup=food_db)
    assert load(spec, judge=JUDGE_A, output_dir=tmp_path).qualitative.aggregate == pytest.approx(0.9)

    _patch_judges(monkeypatch, j1=0.5, j2=0.3)
    forced = await score_runs([spec], judges=(JUDGE_A,), output_dir=tmp_path, lookup=food_db, force=True)
    assert forced[JUDGE_A.key].attempted == 1
    assert forced[JUDGE_A.key].succeeded == 1
    assert load(spec, judge=JUDGE_A, output_dir=tmp_path).qualitative.aggregate == pytest.approx(0.5)


@pytest.mark.asyncio()
async def test_score_runs_single_judge_only_writes_one_folder(
    tmp_path: Path,
    food_db: FoodDb,
    any_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_judges(monkeypatch)
    spec = _run_spec()
    _seed_run(tmp_path, spec, any_code=any_code)

    await score_runs([spec], judges=(JUDGE_B,), output_dir=tmp_path, lookup=food_db)
    assert is_scored(spec, judge=JUDGE_B, output_dir=tmp_path)
    assert not is_scored(spec, judge=JUDGE_A, output_dir=tmp_path)


@pytest.mark.asyncio()
async def test_score_runs_counts_missing_runs(tmp_path: Path, food_db: FoodDb) -> None:
    spec = _run_spec()
    summaries = await score_runs([spec], judges=all_judges(), output_dir=tmp_path, lookup=food_db)
    assert summaries[JUDGE_A.key].missing == 1
    assert summaries[JUDGE_A.key].attempted == 0
    assert not is_scored(spec, judge=JUDGE_A, output_dir=tmp_path)


@pytest.mark.asyncio()
async def test_score_runs_failure_leaves_no_file(
    tmp_path: Path,
    food_db: FoodDb,
    any_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _run_spec()
    _seed_run(tmp_path, spec, any_code=any_code)

    async def boom(*args: object, **kwargs: object) -> QualitativeResult:
        raise RuntimeError("judge down")

    monkeypatch.setattr("evaluation.scoring.score.score_soft_preferences", boom)

    summaries = await score_runs([spec], judges=(JUDGE_A,), output_dir=tmp_path, lookup=food_db)
    assert summaries[JUDGE_A.key].failed == 1
    assert summaries[JUDGE_A.key].succeeded == 0
    assert not is_scored(spec, judge=JUDGE_A, output_dir=tmp_path)
