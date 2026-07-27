"""Critic agent: reviews a proposed `AgentMealPlan`, reports issues, never edits it.

Tool-less by design - it judges only the plan, profile, targets, and
(when the totaller is enabled) the deterministic nutrient totals it is
handed in the prompt, so its critique cannot recurse into its own tool calls.
"""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models import Model

from dietary_advisor.agents.critic.contract import PlanCritique
from dietary_advisor.agents.critic.prompts import critic_agent_system
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.config import get_settings


def build_critic_agent(*, model: Model | None = None, has_user_request: bool = True) -> Agent[AgentDeps, PlanCritique]:
    """Construct a fresh, tool-less critic `Agent` bound to `PlanCritique` output."""
    settings = get_settings()
    return Agent(
        model if model is not None else settings.resolved_llm_model,
        deps_type=AgentDeps,
        output_type=PlanCritique,
        system_prompt=critic_agent_system(has_user_request=has_user_request),
        model_settings=ModelSettings(temperature=settings.llm_temperature),
        retries=2,
    )
