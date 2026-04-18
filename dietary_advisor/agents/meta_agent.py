"""Meta-agent (Agent-Oriented Planning).

In the full V4 pipeline the orchestration is actually performed by
`pipeline.py` (a deterministic Python flow), because deterministic
orchestration is more reproducible for the ablation study than letting an
LLM dispatch tool calls. This module exposes a Meta-Agent only as an
optional alternative orchestrator, in case future work wants to study the
orchestration LLM itself as a moving variable.
"""

from __future__ import annotations

from pydantic_ai import Agent

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.prompts import META_AGENT_SYSTEM
from dietary_advisor.config import get_settings
from dietary_advisor.schemas.meal_plan import MealPlan


def build_meta_agent(model: str | None = None) -> Agent[AgentDeps, MealPlan]:
    settings = get_settings()
    return Agent(
        model or settings.llm_model,
        deps_type=AgentDeps,
        output_type=MealPlan,
        system_prompt=META_AGENT_SYSTEM,
        retries=2,
    )
