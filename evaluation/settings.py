"""Evaluation-only configuration: the G-Eval soft-preference judge model."""

from __future__ import annotations

from functools import cached_property, lru_cache

from pydantic import Field
from pydantic_ai.models import Model

from dietary_advisor.config.settings import Settings


class EvaluationSettings(Settings):
    """Configuration for the `evaluation` ablation harness."""

    # "judge" model used for the G-Eval soft-preference judge
    judge_model: str = Field(...)

    @cached_property
    def resolved_judge_model(self) -> Model:
        return self._resolve(self.judge_model)


@lru_cache(maxsize=1)
def get_evaluation_settings() -> EvaluationSettings:
    s = EvaluationSettings()  # type: ignore[call-arg]
    s.ensure_dirs()
    return s
