"""Run collection: execute the (model, variant, scenario) grid into the output dir.

No scoring lives here - that is the later `evaluate` stage.
"""

from evaluation.case_runner.collect import collect_runs
from evaluation.case_runner.grid import experiment_runs, EXPERIMENTS, planned_runs, RunSpec
from evaluation.case_runner.store import is_done, load, path_for, RunRecord, save

__all__ = [
    "EXPERIMENTS",
    "RunRecord",
    "RunSpec",
    "collect_runs",
    "experiment_runs",
    "is_done",
    "load",
    "path_for",
    "planned_runs",
    "save",
]
