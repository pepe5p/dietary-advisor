"""Execute remaining RunSpecs, skipping any whose result file already exists."""

from __future__ import annotations

import logging
import time
import traceback
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from dietary_advisor.config.llm import LlmSpec
from dietary_advisor.food_db import FoodDb
from dietary_advisor.planning.pipeline import Pipeline, PipelineResult, VariantConfig
from dietary_advisor.profile import UserProfile
from dietary_advisor.profiles import get_profile
from evaluation.case_runner.grid import planned_runs, RunSpec
from evaluation.case_runner.store import is_done, record_from_result, save
from evaluation.scenarios import SCENARIOS
from evaluation.settings import get_evaluation_settings
from evaluation.validation.quantitative import macro_errors

log = logging.getLogger(__name__)

RunFn = Callable[[UserProfile, str], Awaitable[PipelineResult]]


@dataclass(frozen=True)
class CollectSummary:
    planned: int
    already_done: int
    attempted: int
    succeeded: int
    failed: int


async def collect_runs(
    specs: list[RunSpec] | None = None,
    *,
    output_dir: Path | None = None,
    run_fn: RunFn | None = None,
    lookup: FoodDb | None = None,
) -> CollectSummary:
    """Run every unfinished spec; successful results are written under `output_dir`.

    Failures leave no file, so the next invocation retries them. `run_fn` is
    injected only by tests; production always builds a `Pipeline` per
    (model, variant) group.
    """
    dest = output_dir if output_dir is not None else get_evaluation_settings().output_dir
    all_specs = list(specs) if specs is not None else planned_runs()
    remaining = [s for s in all_specs if not is_done(s, output_dir=dest)]
    already_done = len(all_specs) - len(remaining)
    succeeded = 0
    failed = 0

    if not remaining:
        return CollectSummary(
            planned=len(all_specs),
            already_done=already_done,
            attempted=0,
            succeeded=0,
            failed=0,
        )

    owns_lookup = lookup is None
    food_db = lookup if lookup is not None else FoodDb.open()
    try:
        if run_fn is not None:
            for spec in remaining:
                ok = await _run_and_save(spec, run=run_fn, output_dir=dest, food_db=food_db)
                if ok:
                    succeeded += 1
                else:
                    failed += 1
            return CollectSummary(
                planned=len(all_specs),
                already_done=already_done,
                attempted=len(remaining),
                succeeded=succeeded,
                failed=failed,
            )

        groups: dict[tuple[LlmSpec, VariantConfig], list[RunSpec]] = defaultdict(list)
        for spec in remaining:
            groups[(spec.llm, spec.variant)].append(spec)

        for (llm, variant), group in groups.items():
            with Pipeline(variant, food_db=food_db, llm=llm) as pipeline:
                # Default-arg bind: loop rebinds `pipeline` each iteration.
                async def _default_run(
                    profile: UserProfile,
                    query: str,
                    *,
                    _pipeline: Pipeline = pipeline,
                ) -> PipelineResult:
                    return await _pipeline.run(profile, query)

                for spec in group:
                    ok = await _run_and_save(spec, run=_default_run, output_dir=dest, food_db=food_db)
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
    finally:
        if owns_lookup:
            food_db.close()

    return CollectSummary(
        planned=len(all_specs),
        already_done=already_done,
        attempted=len(remaining),
        succeeded=succeeded,
        failed=failed,
    )


async def _run_and_save(
    spec: RunSpec,
    *,
    run: RunFn,
    output_dir: Path,
    food_db: FoodDb,
) -> bool:
    scenario = SCENARIOS[spec.scenario_id]
    profile = get_profile(scenario.profile_id)
    t0 = time.perf_counter()
    try:
        result = await run(profile, scenario.query)
    except Exception as exc:  # noqa: BLE001 - LLM/pipeline failures must not abort the grid
        log.warning(
            "Run failed for %s: %s\n%s",
            spec.spec_key,
            exc,
            traceback.format_exc(limit=4),
        )
        return False

    elapsed = time.perf_counter() - t0
    errors = macro_errors(result.agent_plan, result.targets, food_db)
    record = record_from_result(
        spec=spec,
        query=scenario.query,
        agent_plan=result.agent_plan,
        targets=result.targets,
        iterations=result.iterations,
        telemetry=result.telemetry.as_dict(),
        elapsed_s=round(elapsed, 2),
        errors=errors,
    )
    dest = save(spec=spec, record=record, output_dir=output_dir)
    log.info(
        "Saved %s -> %s (%.1fs)",
        spec.spec_key,
        dest,
        elapsed,
    )
    return True
