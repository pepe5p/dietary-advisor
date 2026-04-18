"""Shared dependency container passed to pydantic-ai agents via `RunContext.deps`.

Holding all collaborators in one immutable bundle keeps the agent tools free
of global lookups and makes it trivial to inject mocks during testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dietary_advisor.knowledge.retriever import HybridRetriever
from dietary_advisor.profile_manager.service import ProfileService
from dietary_advisor.schemas.constraints import HardConstraint
from dietary_advisor.schemas.nutrition import FoodItem, MacroTargets
from dietary_advisor.schemas.profile import UserProfile
from dietary_advisor.tools.usda_client import USDAClient


@dataclass
class AgentDeps:
    """Bundle of services consumed by all specialised agents."""

    profile: UserProfile
    targets: MacroTargets
    constraints: list[HardConstraint] = field(default_factory=list)
    profile_service: ProfileService | None = None
    usda: USDAClient | None = None
    retriever: HybridRetriever | None = None
    # Mutable scratch space the LLM-driven Nutrition agent uses to record the
    # candidate FoodItems it has shortlisted. Centralising it here means the
    # subsequent MILP / validation steps in `pipeline.py` can pick them up.
    shortlist: list[FoodItem] = field(default_factory=list)
