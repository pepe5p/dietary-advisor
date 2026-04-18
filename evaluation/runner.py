"""Batch runner: cross every variant with every scenario and collect metrics.

Produces a tidy `pandas.DataFrame` with one row per
(variant, profile_id, query, repeat) that can be sliced/aggregated freely
in the report module or in a Jupyter notebook.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from dietary_advisor.pipeline import Pipeline, PipelineResult, VariantConfig
from dietary_advisor.profile_manager.service import ProfileService
from dietary_advisor.profile_manager.store import ProfileStore
from evaluation.metrics import csr, faithfulness, hsr, nutrient_errors, ssr
from evaluation.scenarios import Scenario, filter_scenarios

log = logging.getLogger(__name__)


def _ensure_profiles_loaded(profile_dir: Path) -> ProfileService:
    """Load every profile in `profile_dir` into the SQLite store (idempotent)."""
    store = ProfileStore()
    if profile_dir.exists():
        store.import_dir(profile_dir)
    return ProfileService(store=store)


async def _run_one(pipeline: Pipeline, scenario: Scenario, service: ProfileService) -> dict[str, object]:
    profile = service.get(scenario.profile_id)
    if profile is None:
        return {
            "variant": pipeline.variant.name,
            "profile_id": scenario.profile_id,
            "level": scenario.level,
            "error": f"profile {scenario.profile_id} missing",
        }
    t0 = time.perf_counter()
    try:
        result: PipelineResult = await pipeline.run(profile, scenario.query)
    except Exception as exc:  # noqa: BLE001
        log.warning("Run failed for %s/%s: %s", pipeline.variant.name, scenario.profile_id, exc)
        return {
            "variant": pipeline.variant.name,
            "profile_id": scenario.profile_id,
            "level": scenario.level,
            "query": scenario.query,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=4),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
    elapsed = time.perf_counter() - t0

    h = hsr(result.report, result.constraints)
    s = ssr(result.plan, profile)
    c = csr(result.report, result.plan, profile, result.constraints)
    err = nutrient_errors(result.report, result.targets)
    f = faithfulness(result.plan, result.citations or result.plan.citations)

    return {
        "variant": pipeline.variant.name,
        "profile_id": scenario.profile_id,
        "level": scenario.level,
        "query": scenario.query,
        "iterations": result.iterations,
        "n_meals": len(result.plan.meals),
        "n_constraints": len(result.constraints),
        "n_violations": len(result.report.violations),
        "hard_satisfied": result.report.hard_satisfied,
        "HSR": round(h, 4),
        "SSR": round(s, 4),
        "CSR": c,
        "MAE_pct": err.mae,
        "MSE_pct": err.mse,
        **{f"err_{k}_pct": v for k, v in err.per_nutrient.items()},
        "Faithfulness": f,
        "kcal_target": result.targets.energy_kcal,
        "kcal_actual": result.report.totals.get("energy_kcal", 0.0)
        if isinstance(result.report.totals, dict) else 0.0,
        "elapsed_s": round(elapsed, 2),
        "error": None,
    }


async def run_ablation_grid(
    *,
    variants: list[VariantConfig],
    levels: list[int],
    repeats: int = 1,
    profile_dir: Path = Path("evaluation/profiles"),
) -> pd.DataFrame:
    """Run every (variant, scenario, repeat) combination and return a DataFrame."""
    service = _ensure_profiles_loaded(profile_dir)
    scenarios = filter_scenarios(levels)
    rows: list[dict[str, object]] = []

    for variant in variants:
        pipeline = Pipeline(variant, profile_service=service)
        for scenario in scenarios:
            for rep in range(repeats):
                row = await _run_one(pipeline, scenario, service)
                row["repeat"] = rep
                row["variant_description"] = variant.description
                rows.append(row)
                log.info(
                    "Done %s | %s (rep %d) | HSR=%s CSR=%s MAE=%s",
                    variant.name, scenario.profile_id, rep,
                    row.get("HSR"), row.get("CSR"), row.get("MAE_pct"),
                )

    return pd.DataFrame(rows)


def variant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a tidy DataFrame to (variant x level) means."""
    metric_cols = ["HSR", "SSR", "CSR", "MAE_pct", "MSE_pct", "Faithfulness", "iterations", "elapsed_s"]
    cols = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby(["variant", "level"])[cols].mean(numeric_only=True).reset_index()
    return grouped.round(3)


def cli_main() -> None:  # pragma: no cover - convenience helper
    """Allow `python -m evaluation.runner` for ad-hoc runs."""
    from evaluation.ablation import all_variants

    df = asyncio.run(
        run_ablation_grid(variants=all_variants(), levels=[1, 2, 3], repeats=1),
    )
    print(variant_summary(df).to_string(index=False))  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    cli_main()
