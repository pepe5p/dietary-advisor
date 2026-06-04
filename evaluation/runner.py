"""Batch runner: cross every variant with every scenario and collect metrics."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections.abc import Awaitable, Callable

import pandas as pd

from dietary_advisor.pipeline import Pipeline, PipelineResult, VariantConfig
from dietary_advisor.profile_manager.service import ProfileService
from dietary_advisor.schemas.nutrition import NutrientName
from dietary_advisor.schemas.profile import UserProfile
from dietary_advisor.tools.food_lookup import FoodLookup
from dietary_advisor.tools.totaller import total_agent_meal_plan
from dietary_advisor.tools.usda_client import USDAClient
from evaluation.profiles.cases import EVAL_CASES, get_case
from evaluation.profiles.eval_profile import case_complexity, EvalProfile
from evaluation.scenarios import filter_scenarios, Scenario
from evaluation.validation.hydrate import meal_plan_to_eval_plan
from evaluation.validation.qualitative import score_soft_preferences
from evaluation.validation.quantitative import macro_errors
from evaluation.validation.structural import structural_csr

log = logging.getLogger(__name__)

RunFn = Callable[[UserProfile, str], Awaitable[PipelineResult]]


def _profile_service_from_cases() -> ProfileService:
    """Seed the profile store with frozen evaluation cases (idempotent)."""
    service = ProfileService.default()
    for eval_profile in EVAL_CASES.values():
        service.upsert(eval_profile.profile)
    return service


async def _score_row(
    result: PipelineResult,
    scenario: Scenario,
    eval_profile: EvalProfile,
    lookup: FoodLookup,
    *,
    run_judge: bool,
    variant_name: str,
    elapsed_s: float,
) -> dict[str, object]:
    conversion = meal_plan_to_eval_plan(result.plan)
    eval_plan = conversion.plan
    for w in conversion.warnings:
        log.warning("Eval conversion %s: %s", scenario.case_id, w)

    csr = structural_csr(eval_plan, eval_profile, lookup)
    err = macro_errors(eval_plan, eval_profile.macro_targets, lookup)
    nutrient_totals = total_agent_meal_plan(eval_plan, lookup).totals

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
        "level": case_complexity(scenario.case_id),
        "query": scenario.query,
        "iterations": result.iterations,
        "n_meals": len(eval_plan.meals),
        "n_constraints": len(eval_profile.hard_constraints),
        "n_violations": len(result.report.violations),
        "hard_satisfied": result.report.hard_satisfied,
        "CSR": csr,
        "MAE_pct": err.mae,
        "MSE_pct": err.mse,
        **{f"err_{k}_pct": v for k, v in err.per_nutrient.items()},
        "SoftScore": soft_score,
        "SoftDetail": soft_detail,
        "kcal_target": eval_profile.macro_targets.energy_kcal,
        "kcal_actual": float(nutrient_totals.get(NutrientName.ENERGY_KCAL, 0.0)),
        "conversion_warnings": len(conversion.warnings),
        "elapsed_s": round(elapsed_s, 2),
        "error": None,
    }


async def _run_one(
    *,
    scenario: Scenario,
    eval_profile: EvalProfile,
    lookup: FoodLookup,
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
            "level": case_complexity(scenario.case_id),
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
    lookup: FoodLookup,
    *,
    variants: list[VariantConfig],
    scenarios: list[Scenario],
    repeats: int,
    run_judge: bool,
    run_fn: RunFn | None,
    service: ProfileService,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        with Pipeline(variant, profile_service=service) as pipeline:

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
                        variant_name=variant.name,
                        run=run,
                    )
                    row["repeat"] = rep
                    row["variant_description"] = variant.description
                    rows.append(row)
                    log.info(
                        "Done %s | %s (rep %d) | CSR=%s MAE=%s Soft=%s",
                        variant.name,
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
    levels: list[int],
    repeats: int = 1,
    run_judge: bool = True,
    run_fn: RunFn | None = None,
    lookup: FoodLookup | None = None,
) -> pd.DataFrame:
    """Run every (variant, scenario, repeat) combination and return a DataFrame."""
    service = _profile_service_from_cases()
    scenarios = filter_scenarios(levels)

    if lookup is not None:
        rows = await _run_ablation_grid_with_lookup(
            lookup,
            variants=variants,
            scenarios=scenarios,
            repeats=repeats,
            run_judge=run_judge,
            run_fn=run_fn,
            service=service,
        )
        return pd.DataFrame(rows)

    with USDAClient() as client:
        rows = await _run_ablation_grid_with_lookup(
            client,
            variants=variants,
            scenarios=scenarios,
            repeats=repeats,
            run_judge=run_judge,
            run_fn=run_fn,
            service=service,
        )
    return pd.DataFrame(rows)


def variant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a tidy DataFrame to (variant x level) means."""
    metric_cols = ["CSR", "MAE_pct", "MSE_pct", "SoftScore", "iterations", "elapsed_s"]
    cols = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby(["variant", "level"])[cols].mean(numeric_only=True).reset_index()
    return grouped.round(3)


def cli_main() -> None:  # pragma: no cover
    from evaluation.ablation import all_variants

    df = asyncio.run(
        run_ablation_grid(variants=all_variants(), levels=[1, 2, 3], repeats=1),
    )
    print(variant_summary(df).to_string(index=False))  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    cli_main()
