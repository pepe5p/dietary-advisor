"""Critic agent package: reviews a proposed plan, reports issues, never edits it."""

from dietary_advisor.agents.critic.agent import build_critic_agent
from dietary_advisor.agents.critic.contract import PlanCritique

__all__ = [
    "PlanCritique",
    "build_critic_agent",
]
