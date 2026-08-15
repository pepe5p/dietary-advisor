"""G-Eval judge registry.

Adding a judge is a single line in ``JUDGES_BY_KEY``: every consumer iterates the
registry instead of naming individual judges, so scores, figures, metrics and
LaTeX columns all follow automatically. All judges share one rubric, so the only
thing that differs between them is the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.models import infer_model, Model


def resolve_judge_model(model_id: str) -> Model:
    return infer_model(model_id)


@dataclass(frozen=True)
class Judge:
    key: str
    model_id: str

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

# Each judge scores every run this many times; reporting uses the mean verdict.
JUDGE_REPS = 3


def all_judges() -> tuple[Judge, ...]:
    return tuple(JUDGES_BY_KEY.values())
