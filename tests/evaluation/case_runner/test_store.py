"""Tests for case-runner JSON store helpers."""

from __future__ import annotations

from pathlib import Path

from dietary_advisor.config.llm import LlmSpec
from dietary_advisor.planning.pipeline import VariantConfig
from dietary_advisor.totaller.nutrition import MacroTargets
from evaluation.case_runner.grid import RunSpec
from evaluation.case_runner.store import is_done, load, path_for, RunRecord, save
from tests.evaluation.conftest import agent_plan_rice_lunch


def test_path_for_sanitizes_model_id(tmp_path: Path) -> None:
    spec = RunSpec(
        llm=LlmSpec(model="groq:llama-3.3-70b-versatile"),
        variant=VariantConfig(),
        scenario_id="regular",
    )
    path = path_for(spec, output_dir=tmp_path)
    assert path.parent == tmp_path / "runs"
    assert path.name == "groq-llama-3.3-70b-versatile__totaller+reflective-loop__regular__rep0.json"


def test_save_load_round_trip(tmp_path: Path) -> None:
    variant = VariantConfig(rag_enabled=False)
    spec = RunSpec(llm=LlmSpec(model="gemini-3.1-flash-lite"), variant=variant, scenario_id="regular")
    record = RunRecord(
        llm_model=str(spec.llm),
        variant=variant.label,
        totaller_enabled=variant.totaller_enabled,
        rag_enabled=variant.rag_enabled,
        reflection_enabled=variant.reflection_enabled,
        scenario_id=spec.scenario_id,
        query="Plan one balanced day.",
        agent_plan=agent_plan_rice_lunch(user_id="regular"),
        targets=MacroTargets(energy_kcal=2000, protein_g=100, carbs_g=200, fat_g=70),
        mae_pct=5.5,
        mse_pct=42.0,
        per_nutrient_pct={"energy_kcal": 3.0},
        iterations=1,
        telemetry={"requests": 2, "total_tokens": 100},
        elapsed_s=1.5,
    )
    dest = save(spec=spec, record=record, output_dir=tmp_path)
    assert dest.is_file()
    assert is_done(spec=spec, output_dir=tmp_path)

    loaded = load(spec=spec, output_dir=tmp_path)
    assert loaded.llm_model == record.llm_model
    assert loaded.variant == "totaller+reflective-loop"
    assert loaded.scenario_id == "regular"
    assert loaded.agent_plan.user_id == "regular"
    assert loaded.targets.energy_kcal == 2000
    assert loaded.iterations == 1
    assert loaded.mae_pct == 5.5
    assert loaded.mse_pct == 42.0
    assert loaded.per_nutrient_pct == {"energy_kcal": 3.0}
