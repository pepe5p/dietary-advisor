"""Critic output for the reflection loop: a list of concrete issues, or none."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlanCritique(BaseModel):
    """Structured verdict from the critic agent on a proposed `AgentMealPlan`."""

    model_config = ConfigDict(extra="forbid")

    issues: list[str] = Field(default_factory=list)
