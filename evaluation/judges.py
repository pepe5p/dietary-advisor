"""G-Eval judge registry.

Adding a judge is a single line in ``JUDGES_BY_KEY``: every consumer iterates the
registry instead of naming individual judges, so scores, figures, metrics and
LaTeX columns all follow automatically. All judges share one rubric, so the only
thing that differs between them is the model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai.models import infer_model, Model


def _shared_judge_system_prompt(*, has_user_query: bool = True) -> str:
    from evaluation.validation.qualitative import judge_soft_preferences_system

    return judge_soft_preferences_system(has_user_query=has_user_query)


def resolve_judge_model(model_id: str) -> Model:
    return infer_model(model_id)


@dataclass(frozen=True)
class Judge:
    key: str
    model_id: str
    system_prompt: Callable[..., str] = field(default=_shared_judge_system_prompt)

    @property
    def label(self) -> str:
        """Compact display label for figure legends and table headers."""
        return self.key.replace("judge_", "J")

    @property
    def scores_subdir(self) -> str:
        return f"scores_{self.key}"

    @property
    def resolved_model(self) -> Model:
        return resolve_judge_model(self.model_id)


JUDGES_BY_KEY = {
    "judge_1": Judge("judge_1", "openrouter:openai/gpt-5.6-luna"),
    "judge_2": Judge("judge_2", "openrouter:deepseek/deepseek-v4-flash-0731"),
}


def all_judges() -> tuple[Judge, ...]:
    return tuple(JUDGES_BY_KEY.values())
