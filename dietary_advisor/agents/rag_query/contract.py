"""RAG query schema: the search queries the query agent derives from a profile.

Output of the RAG query agent (see `dietary_advisor.agents.rag_query`),
a tool-less brainstorming pass that turns the user's request and profile into
targeted clinical-guideline search queries before any retrieval happens. The
upper bound exists only to cap retrieval cost - the agent is told to scale the
count with profile complexity, not to hit a fixed number.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievalQueries(BaseModel):
    """A set of clinical-guideline search queries for one recommendation request."""

    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1, max_length=10)
