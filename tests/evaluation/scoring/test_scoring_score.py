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
from evaluation.scoring.score import score_runs
from evaluation.scoring.store import is_scored, load
from evaluation.settings import get_evaluation_settings
from evaluation.validation.qualitative import CriterionScore, QualitativeResult
from tests.evaluation.conftest import agent_plan_single


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


def _patch_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = QualitativeResult(
        scores=[
            CriterionScore(
                criterion_id="recipe-makes-sense",
                score=0.9,
                reasoning="Coherent.",
            ),
        ],
        aggregate=0.9,
    )
    monkeypatch.setitem(
        get_evaluation_settings().__dict__,
        "resolved_judge_model",
        TestModel(custom_output_args=expected.model_dump()),
    )


@pytest.mark.asyncio()
async def test_score_runs_writes_and_skips_existing(
    tmp_path: Path,
    food_db: FoodDb,
    any_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_judge(monkeypatch)
    spec = _run_spec()
    _seed_run(tmp_path, spec, any_code=any_code)

    first = await score_runs([spec], output_dir=tmp_path, lookup=food_db)
    assert first.succeeded == 1
    assert first.failed == 0
    assert is_scored(spec, output_dir=tmp_path)

    second = await score_runs([spec], output_dir=tmp_path, lookup=food_db)
    assert second.already_scored == 1
    assert second.attempted == 0


@pytest.mark.asyncio()
async def test_score_runs_force_rescores(
    tmp_path: Path,
    food_db: FoodDb,
    any_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_judge(monkeypatch)
    spec = _run_spec()
    _seed_run(tmp_path, spec, any_code=any_code)

    await score_runs([spec], output_dir=tmp_path, lookup=food_db)
    first = load(spec, output_dir=tmp_path)
    assert first.qualitative.aggregate == pytest.approx(0.9)

    expected = QualitativeResult(
        scores=[
            CriterionScore(
                criterion_id="recipe-makes-sense",
                score=0.5,
                reasoning="Weaker.",
            ),
        ],
        aggregate=0.5,
    )
    monkeypatch.setitem(
        get_evaluation_settings().__dict__,
        "resolved_judge_model",
        TestModel(custom_output_args=expected.model_dump()),
    )

    forced = await score_runs([spec], output_dir=tmp_path, lookup=food_db, force=True)
    assert forced.attempted == 1
    assert forced.succeeded == 1
    reloaded = load(spec, output_dir=tmp_path)
    assert reloaded.qualitative.aggregate == pytest.approx(0.5)


@pytest.mark.asyncio()
async def test_score_runs_counts_missing_runs(tmp_path: Path, food_db: FoodDb) -> None:
    spec = _run_spec()
    summary = await score_runs([spec], output_dir=tmp_path, lookup=food_db)
    assert summary.missing == 1
    assert summary.attempted == 0
    assert not is_scored(spec, output_dir=tmp_path)


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

    summary = await score_runs([spec], output_dir=tmp_path, lookup=food_db)
    assert summary.failed == 1
    assert summary.succeeded == 0
    assert not is_scored(spec, output_dir=tmp_path)
