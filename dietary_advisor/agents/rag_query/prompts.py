"""RAG query agent prompts: system prompt and user-turn composer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dietary_advisor.agents.prompt_blocks import has_user_request, join_sections

if TYPE_CHECKING:
    from dietary_advisor.agents.deps import AgentDeps


_RAG_QUERY_INTRO = """You are a clinical-guideline search strategist. Given {subject}, produce the
search queries that will retrieve the specialized clinical nutrition guidance
needed to plan their meals safely.

Phrase each query in English, in the vocabulary of clinical guidelines, not as
a chat question: "sodium intake hypertension", "low glycemic index
carbohydrates type 2 diabetes", "protein requirements chronic kidney disease" -
not "what should someone with high blood pressure eat?".
"""

_RAG_QUERY_TASK_WITH_REQUEST = "Produce the clinical-guideline search queries for this request."
_RAG_QUERY_TASK_NO_REQUEST = "Produce the clinical-guideline search queries for this profile."


def rag_query_agent_system(*, has_user_request: bool = True) -> str:
    subject = "a user's dietary request and profile" if has_user_request else "a user's profile"
    return _RAG_QUERY_INTRO.format(subject=subject)


# Backward-compatible alias for imports that expect a module-level constant.
RAG_QUERY_AGENT_SYSTEM = rag_query_agent_system()


def rag_query_user_prompt(deps: AgentDeps, user_query: str) -> str:
    """The query agent plans retrieval, so it sees the profile but no macro targets."""
    sections: list[str] = []
    if has_user_request(user_query):
        sections.append(f"User request: {user_query}")
    sections.extend(
        [
            "Profile:",
            deps.profile.model_dump_json(indent=2),
            _RAG_QUERY_TASK_WITH_REQUEST if has_user_request(user_query) else _RAG_QUERY_TASK_NO_REQUEST,
        ],
    )
    return join_sections(sections)
