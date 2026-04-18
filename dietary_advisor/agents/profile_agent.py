"""Profile agent: read-only access to the SQLite profile store.

Kept intentionally small - profile mutation is a CLI-only operation.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.prompts import PROFILE_AGENT_SYSTEM
from dietary_advisor.config import get_settings
from dietary_advisor.schemas.profile import UserProfile


def build_profile_agent(model: str | None = None) -> Agent[AgentDeps, UserProfile]:
    settings = get_settings()
    agent: Agent[AgentDeps, UserProfile] = Agent(
        model or settings.llm_model,
        deps_type=AgentDeps,
        output_type=UserProfile,
        system_prompt=PROFILE_AGENT_SYSTEM,
        retries=1,
    )

    @agent.tool
    async def get_profile(ctx: RunContext[AgentDeps], user_id: str) -> UserProfile:
        if ctx.deps.profile_service is None:
            raise RuntimeError("profile_service is not configured.")
        profile = ctx.deps.profile_service.get(user_id)
        if profile is None:
            raise ValueError(f"Unknown user_id: {user_id}")
        return profile

    return agent
