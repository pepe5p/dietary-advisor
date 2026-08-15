"""Score stored case-run records with quantitative and qualitative validators."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dietary_advisor.food_db import FoodDb
from dietary_advisor.profiles import get_profile
from evaluation.case_runner.grid import planned_runs, RunSpec
from evaluation.case_runner.store import is_done
from evaluation.case_runner.store import load as load_run
from evaluation.judges import all_judges, Judge
from evaluation.scenarios import SCENARIOS
from evaluation.scoring.store import is_scored, save, ScoreRecord
from evaluation.settings import get_evaluation_settings
from evaluation.validation.qualitative import score_soft_preferences
from evaluation.validation.quantitative import macro_errors

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreSummary:
    planned: int
    missing: int
    already_scored: int
    attempted: int
    succeeded: int
    failed: int


async def _score_runs_for_judge(
    all_specs: list[RunSpec],
    *,
    judge: Judge,
    output_dir: Path,
    food_db: FoodDb,
    force: bool,
) -> ScoreSummary:
    missing = [s for s in all_specs if not is_done(s, output_dir=output_dir)]
    present = [s for s in all_specs if is_done(s, output_dir=output_dir)]
    already_scored = [s for s in present if is_scored(s, judge=judge, output_dir=output_dir) and not force]
    to_score = [s for s in present if force or not is_scored(s, judge=judge, output_dir=output_dir)]

    if not to_score:
        return ScoreSummary(
            planned=len(all_specs),
            missing=len(missing),
            already_scored=len(already_scored),
            attempted=0,
            succeeded=0,
            failed=0,
        )

    succeeded = 0
    failed = 0
    for spec in to_score:
        ok = await _score_one(spec, food_db=food_db, judge=judge, output_dir=output_dir)
        if ok:
            succeeded += 1
        else:
            failed += 1

    return ScoreSummary(
        planned=len(all_specs),
        missing=len(missing),
        already_scored=len(already_scored),
        attempted=len(to_score),
        succeeded=succeeded,
        failed=failed,
    )


async def score_runs(
    specs: list[RunSpec] | None = None,
    *,
    judges: Sequence[Judge] | None = None,
    output_dir: Path | None = None,
    lookup: FoodDb | None = None,
    force: bool = False,
) -> dict[str, ScoreSummary]:
    """Score every planned spec that has a stored run record, once per judge.

    Already-scored specs are skipped unless ``force`` is set. Failures leave no
    score file, so the next invocation retries them.
    """
    dest = output_dir if output_dir is not None else get_evaluation_settings().output_dir
    all_specs = list(specs) if specs is not None else planned_runs()
    selected = tuple(judges) if judges is not None else all_judges()

    owns_lookup = lookup is None
    food_db = lookup if lookup is not None else FoodDb.open()
    summaries: dict[str, ScoreSummary] = {}
    try:
        for judge in selected:
            summaries[judge.key] = await _score_runs_for_judge(
                all_specs,
                judge=judge,
                output_dir=dest,
                food_db=food_db,
                force=force,
            )
    finally:
        if owns_lookup:
            food_db.close()

    return summaries


async def _score_one(
    spec: RunSpec,
    *,
    food_db: FoodDb,
    judge: Judge,
    output_dir: Path,
) -> bool:
    run_record = load_run(spec, output_dir=output_dir)
    scenario = SCENARIOS[run_record.scenario_id]
    profile = get_profile(scenario.profile_id)
    try:
        errors = macro_errors(run_record.agent_plan, run_record.targets, food_db)
        qualitative = await score_soft_preferences(
            run_record.agent_plan,
            run_record.query,
            scenario.soft_criteria,
            allergens=profile.allergens,
            diet_pattern=profile.diet_pattern,
            judge=judge,
        )
    except Exception as exc:  # noqa: BLE001 - judge/DB failures must not abort the grid
        log.warning(
            "Scoring failed for %s (%s): %s\n%s",
            spec.spec_key,
            judge.key,
            exc,
            traceback.format_exc(limit=4),
        )
        return False

    score_record = ScoreRecord(
        llm_model=run_record.llm_model,
        variant=run_record.variant,
        scenario_id=run_record.scenario_id,
        judge_model=judge.model_id,
        mae_pct=errors.mae,
        mse_pct=errors.mse,
        per_nutrient_pct=errors.per_nutrient,
        qualitative=qualitative,
        iterations=run_record.iterations,
        elapsed_s=run_record.elapsed_s,
    )
    dest = save(spec=spec, record=score_record, judge=judge, output_dir=output_dir)
    log.info(
        "Scored %s (%s) -> %s (MAE %.1f%%, soft %.2f)",
        spec.spec_key,
        judge.key,
        dest,
        errors.mae,
        qualitative.aggregate,
    )
    return True
