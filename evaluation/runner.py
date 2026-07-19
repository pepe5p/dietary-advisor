"""Batch runner: cross every variant with every scenario and collect metrics."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections.abc import Awaitable, Callable

import pandas as pd

from dietary_advisor.food_db import FoodDb
from dietary_advisor.planning.pipeline import Pipeline, PipelineResult, VariantConfig
from dietary_advisor.schemas.agent_output import AgentMealPlan
from dietary_advisor.schemas.nutrition import NutrientName
from dietary_advisor.schemas.profile import UserProfile
from evaluation.profiles.cases import get_case
from evaluation.profiles.eval_profile import EvalProfile
from evaluation.scenarios import Scenario, SCENARIOS
from evaluation.validation.hydrate import hydrate_meal_plan, total_agent_meal_plan
from evaluation.validation.qualitative import score_soft_preferences
from evaluation.validation.quantitative import macro_errors
from evaluation.validation.structural import structural_csr
from evaluation.validation.validator import validate_meal_plan

log = logging.getLogger(__name__)

RunFn = Callable[[UserProfile, str], Awaitable[PipelineResult]]


def _hard_constraint_result(
    eval_plan: AgentMealPlan,
    eval_profile: EvalProfile,
    lookup: FoodDb,
) -> tuple[int, bool]:
    """Hydrate + run the ground-truth Validator, returning (n_violations, hard_satisfied)."""
    try:
        hydrated = hydrate_meal_plan(eval_plan, lookup)
    except Exception as exc:  # noqa: BLE001
        log.warning("Hydration failed while validating hard constraints for %s: %s", eval_profile.case_id, exc)
        return 0, False
    report = validate_meal_plan(hydrated, list(eval_profile.hard_constraints))
    return len(report.violations), report.hard_satisfied


async def _score_row(
    result: PipelineResult,
    scenario: Scenario,
    eval_profile: EvalProfile,
    lookup: FoodDb,
    *,
    run_judge: bool,
    variant_name: str,
    elapsed_s: float,
) -> dict[str, object]:
    eval_plan = result.agent_plan
    csr = structural_csr(eval_plan, eval_profile, lookup)
    err = macro_errors(eval_plan, eval_profile.profile.targets, lookup)
    nutrient_totals = total_agent_meal_plan(eval_plan, lookup).totals
    n_violations, hard_satisfied = _hard_constraint_result(eval_plan, eval_profile, lookup)

    soft_score: float | None = None
    soft_detail: str | None = None
    if run_judge and scenario.soft_criteria:
        qual = await score_soft_preferences(
            eval_plan,
            scenario.query,
            scenario.soft_criteria,
        )
        if qual is not None:
            soft_score = qual.aggregate
            soft_detail = qual.model_dump_json()

    return {
        "variant": variant_name,
        "case_id": scenario.case_id,
        "query": scenario.query,
        "iterations": result.iterations,
        "n_meals": len(eval_plan.meals),
        "n_constraints": len(eval_profile.hard_constraints),
        "n_violations": n_violations,
        "hard_satisfied": hard_satisfied,
        "CSR": csr,
        "MAE_pct": err.mae,
        "MSE_pct": err.mse,
        **{f"err_{k}_pct": v for k, v in err.per_nutrient.items()},
        "SoftScore": soft_score,
        "SoftDetail": soft_detail,
        "kcal_target": eval_profile.profile.targets.energy_kcal,
        "kcal_actual": float(nutrient_totals.get(NutrientName.ENERGY_KCAL, 0.0)),
        "elapsed_s": round(elapsed_s, 2),
        "error": None,
    }


async def _run_one(
    *,
    scenario: Scenario,
    eval_profile: EvalProfile,
    lookup: FoodDb,
    run_judge: bool,
    variant_name: str,
    run: RunFn,
) -> dict[str, object]:
    t0 = time.perf_counter()
    try:
        result = await run(eval_profile.profile, scenario.query)
    except Exception as exc:  # noqa: BLE001
        log.warning("Run failed for %s/%s: %s", variant_name, scenario.case_id, exc)
        return {
            "variant": variant_name,
            "case_id": scenario.case_id,
            "query": scenario.query,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=4),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
    elapsed = time.perf_counter() - t0
    return await _score_row(
        result,
        scenario,
        eval_profile,
        lookup,
        run_judge=run_judge,
        variant_name=variant_name,
        elapsed_s=elapsed,
    )


async def _run_ablation_grid_with_lookup(
    lookup: FoodDb,
    *,
    variants: list[VariantConfig],
    scenarios: list[Scenario],
    repeats: int,
    run_judge: bool,
    run_fn: RunFn | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        with Pipeline(variant) as pipeline:

            async def _default_run(profile: UserProfile, query: str) -> PipelineResult:
                return await pipeline.run(profile, query)

            run: RunFn = run_fn if run_fn is not None else _default_run

            for scenario in scenarios:
                eval_profile = get_case(scenario.case_id)
                for rep in range(repeats):
                    row = await _run_one(
                        scenario=scenario,
                        eval_profile=eval_profile,
                        lookup=lookup,
                        run_judge=run_judge,
                        variant_name=variant.label,
                        run=run,
                    )
                    row["repeat"] = rep
                    row["variant_description"] = variant.description
                    rows.append(row)
                    log.info(
                        "Done %s | %s (rep %d) | CSR=%s MAE=%s Soft=%s",
                        variant.label,
                        scenario.case_id,
                        rep,
                        row.get("CSR"),
                        row.get("MAE_pct"),
                        row.get("SoftScore"),
                    )
    return rows


async def run_ablation_grid(
    *,
    variants: list[VariantConfig],
    repeats: int = 1,
    run_judge: bool = True,
    run_fn: RunFn | None = None,
    lookup: FoodDb | None = None,
) -> pd.DataFrame:
    """Run every (variant, scenario, repeat) combination and return a DataFrame."""
    if lookup is not None:
        rows = await _run_ablation_grid_with_lookup(
            lookup,
            variants=variants,
            scenarios=SCENARIOS,
            repeats=repeats,
            run_judge=run_judge,
            run_fn=run_fn,
        )
        return pd.DataFrame(rows)

    with FoodDb.open() as client:
        rows = await _run_ablation_grid_with_lookup(
            client,
            variants=variants,
            scenarios=SCENARIOS,
            repeats=repeats,
            run_judge=run_judge,
            run_fn=run_fn,
        )
    return pd.DataFrame(rows)


def variant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a tidy DataFrame to per-variant means."""
    metric_cols = ["CSR", "MAE_pct", "MSE_pct", "SoftScore", "iterations", "elapsed_s"]
    cols = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby(["variant"])[cols].mean(numeric_only=True).reset_index()
    return grouped.round(3)


def cli_main() -> None:  # pragma: no cover
    from evaluation.ablation import leave_one_out_variants

    df = asyncio.run(
        run_ablation_grid(variants=leave_one_out_variants(), repeats=1),
    )
    print(variant_summary(df).to_string(index=False))  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    cli_main()
