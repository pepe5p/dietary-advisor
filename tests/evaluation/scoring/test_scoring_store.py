"""Tests for score JSON store helpers."""

from __future__ import annotations

from pathlib import Path

from dietary_advisor.planning.pipeline import VariantConfig
from evaluation.case_runner.grid import RunSpec
from evaluation.scoring.store import is_scored, load, path_for, save, ScoreRecord
from evaluation.validation.qualitative import CriterionScore, QualitativeResult


def test_path_for_uses_scores_subdir(tmp_path: Path) -> None:
    spec = RunSpec(
        llm_model="groq:llama-3.3-70b-versatile",
        variant=VariantConfig(),
        scenario_id="regular",
    )
    path = path_for(spec, output_dir=tmp_path)
    assert path.parent == tmp_path / "scores"
    assert path.name == "groq-llama-3.3-70b-versatile__totaller+reflective-loop__regular__rep0.json"


def test_save_load_round_trip(tmp_path: Path) -> None:
    spec = RunSpec(llm_model="test-model", variant=VariantConfig(), scenario_id="regular")
    record = ScoreRecord(
        llm_model=spec.llm_model,
        variant="totaller+reflective-loop",
        scenario_id="regular",
        judge_model="gemini-3.5-flash-lite",
        mae_pct=5.5,
        mse_pct=42.0,
        per_nutrient_pct={"energy_kcal": 3.0},
        qualitative=QualitativeResult(
            scores=[
                CriterionScore(
                    criterion_id="recipe-makes-sense",
                    score=0.9,
                    reasoning="Looks fine.",
                ),
            ],
            aggregate=0.9,
        ),
        iterations=1,
        elapsed_s=2.5,
    )
    dest = save(spec=spec, record=record, output_dir=tmp_path)
    assert dest.is_file()
    assert is_scored(spec=spec, output_dir=tmp_path)

    loaded = load(spec=spec, output_dir=tmp_path)
    assert loaded.mae_pct == record.mae_pct
    assert loaded.qualitative.aggregate == record.qualitative.aggregate
