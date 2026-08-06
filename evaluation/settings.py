"""Evaluation-only configuration: judge model and case-run output directory."""

from __future__ import annotations

from functools import cached_property, lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_ai.models import infer_model, Model

from dietary_advisor.config.settings import Settings

_REPO_ROOT = Path(__file__).resolve().parent.parent


class EvaluationSettings(Settings):
    """Configuration for the `evaluation` ablation harness."""

    # "judge" model used for the G-Eval soft-preference judge
    judge_model: str = Field(...)
    # Successful case-run JSON files land here (project-top-level `outputs/` by default).
    output_dir: Path = Field(default=_REPO_ROOT / "outputs")

    @cached_property
    def resolved_judge_model(self) -> Model:
        return infer_model(self.judge_model)

    def ensure_dirs(self) -> None:
        super().ensure_dirs()
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_evaluation_settings() -> EvaluationSettings:
    s = EvaluationSettings()  # type: ignore[call-arg]
    s.ensure_dirs()
    return s
