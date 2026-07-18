"""RAG agent: owns the hybrid retriever tool.

Runs ahead of `nutrition_agent` to gather grounded clinical evidence for the
patient's specific condition profile.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, ModelSettings, RunContext

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.prompts import RAG_AGENT_SYSTEM
from dietary_advisor.config import get_settings


def build_rag_agent() -> Agent[AgentDeps, list[dict[str, Any]]]:
    settings = get_settings()
    agent: Agent[AgentDeps, list[dict[str, Any]]] = Agent(
        settings.resolved_llm_model,
        deps_type=AgentDeps,
        output_type=list[dict[str, Any]],
        system_prompt=RAG_AGENT_SYSTEM,
        model_settings=ModelSettings(temperature=settings.llm_temperature),
        retries=1,
    )

    @agent.tool
    async def retrieve(ctx: RunContext[AgentDeps], query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if ctx.deps.retriever is None:
            return []
        chunks = ctx.deps.retriever.retrieve(query, top_k=top_k)
        return [
            {
                "id": c.id,
                "text": c.text,
                "doc_id": c.metadata.get("doc_id"),
                "title": c.metadata.get("title"),
                "page": c.metadata.get("page"),
                "score": c.score,
            }
            for c in chunks
        ]

    return agent
