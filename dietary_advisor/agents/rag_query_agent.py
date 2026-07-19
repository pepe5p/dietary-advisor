"""RAG query agent: derives clinical-guideline search queries from a profile.

Runs as the first pipeline step, ahead of retrieval, so the queries fed to the
retriever are shaped by the user's request and their full profile (conditions,
diet pattern, allergens) rather than a fixed string concatenation. Like the
meal-idea agent it owns no tools and its output is not DB-verified; the
retriever simply runs whatever queries it returns.
"""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.prompts import RAG_QUERY_AGENT_SYSTEM
from dietary_advisor.config import get_settings
from dietary_advisor.schemas.rag_query import RetrievalQueries


def build_rag_query_agent() -> Agent[AgentDeps, RetrievalQueries]:
    settings = get_settings()
    return Agent(
        settings.resolved_llm_model,
        deps_type=AgentDeps,
        output_type=RetrievalQueries,
        system_prompt=RAG_QUERY_AGENT_SYSTEM,
        model_settings=ModelSettings(temperature=settings.llm_temperature),
        retries=1,
    )
