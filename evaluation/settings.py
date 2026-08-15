"""Evaluation-only configuration: case-run output directory."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field

from dietary_advisor.config.settings import Settings

_REPO_ROOT = Path(__file__).resolve().parent.parent


class EvaluationSettings(Settings):
    """Configuration for the `evaluation` ablation harness."""

    # Successful case-run JSON files land here (project-top-level `outputs/` by default).
    output_dir: Path = Field(default=_REPO_ROOT / "outputs")

    def ensure_dirs(self) -> None:
        super().ensure_dirs()
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_evaluation_settings() -> EvaluationSettings:
    s = EvaluationSettings()  # type: ignore[call-arg]
    s.ensure_dirs()
    return s
