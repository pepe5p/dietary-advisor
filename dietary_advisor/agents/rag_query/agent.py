"""RAG query agent: derives clinical-guideline search queries from a profile.

Runs as the first pipeline step, ahead of retrieval, so the queries fed to the
retriever are shaped by the user's request and their full profile (conditions,
diet pattern, allergens) rather than a fixed string concatenation. Like the
meal-idea agent it owns no tools and its output is not DB-verified; the
retriever simply runs whatever queries it returns.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.rag_query.contract import RetrievalQueries
from dietary_advisor.agents.rag_query.prompts import rag_query_agent_system
from dietary_advisor.config import get_settings


def build_rag_query_agent(
    *,
    model: Model | None = None,
    has_user_request: bool = True,
) -> Agent[AgentDeps, RetrievalQueries]:
    settings = get_settings()
    return Agent(
        model if model is not None else settings.resolved_llm_model,
        deps_type=AgentDeps,
        output_type=RetrievalQueries,
        system_prompt=rag_query_agent_system(has_user_request=has_user_request),
        retries=1,
    )
