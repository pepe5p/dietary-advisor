"""Shared dependency container passed to pydantic-ai agents via `RunContext.deps`.

Holding all collaborators in one immutable bundle keeps the agent tools free
of global lookups and makes it trivial to inject mocks during testing.
"""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.dietary_rag.retriever import HybridRetriever
from dietary_advisor.food_db import FoodDb
from dietary_advisor.schemas.nutrition import MacroTargets
from dietary_advisor.schemas.profile import UserProfile


@dataclass
class AgentDeps:
    """Bundle of services consumed by all specialised agents."""

    profile: UserProfile
    targets: MacroTargets
    food_db: FoodDb
    retriever: HybridRetriever | None = None
