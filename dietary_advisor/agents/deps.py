"""Shared dependency container passed to pydantic-ai agents via `RunContext.deps`.

Holding all collaborators in one immutable bundle keeps the agent tools free
of global lookups and makes it trivial to inject mocks during testing.
"""

from __future__ import annotations

from dataclasses import dataclass

from dietary_advisor.knowledge.retriever import HybridRetriever
from dietary_advisor.schemas.nutrition import MacroTargets
from dietary_advisor.schemas.profile import UserProfile
from dietary_advisor.tools.food_db import OffFoodDb


@dataclass
class AgentDeps:
    """Bundle of services consumed by all specialised agents.

    Deliberately carries no constraints: production never validates against
    hard constraints (that is an evaluation-only concept) - the agent must
    infer restrictions from `profile` itself.
    """

    profile: UserProfile
    targets: MacroTargets
    food_db: OffFoodDb | None = None
    retriever: HybridRetriever | None = None
