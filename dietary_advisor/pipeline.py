"""Top-level orchestrator with feature flags for the V0-V4 ablation variants.

The pipeline is intentionally a *deterministic Python flow* rather than an
LLM-driven Meta-Agent: that keeps the variant comparison clean (no second
hidden LLM behaviour to worry about). Each feature flag toggles exactly one
module, mapping 1:1 to the four thesis modules.

Variants:
    V0 - Baseline:      bare LLM, no tools, no RAG, no validator
    V1 - +Totaller:     LLM + deterministic totaller + USDA-grounded foods
    V2 - +Profile:      V1 + profile-derived hard constraints (allergens, ...)
    V3 - +RAG:          V2 + clinical-guideline retrieval + condition rules
    V4 - +Reflection:   V3 + Generate-Score-Refine loop (full system)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition_agent import build_nutrition_agent
from dietary_advisor.config import get_settings
from dietary_advisor.knowledge.retriever import HybridRetriever
from dietary_advisor.profile_manager.service import ProfileService
from dietary_advisor.schemas.constraints import HardConstraint, ValidationReport
from dietary_advisor.schemas.meal_plan import Citation, MealPlan
from dietary_advisor.schemas.nutrition import MacroTargets
from dietary_advisor.schemas.profile import UserProfile
from dietary_advisor.tools.tdee import derive_macro_targets
from dietary_advisor.tools.totaller import total_meal_plan
from dietary_advisor.tools.usda_client import USDAClient
from dietary_advisor.validation.reflection import reflect_and_refine, ReflectionResult
from dietary_advisor.validation.validator import validate_meal_plan

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantConfig:
    """Feature flags that distinguish a variant from the V0 baseline."""

    name: str
    totaller_enabled: bool = False
    profile_constraints_enabled: bool = False
    rag_enabled: bool = False
    reflection_enabled: bool = False
    description: str = ""

    @property
    def needs_usda(self) -> bool:
        return self.totaller_enabled


VARIANTS: dict[str, VariantConfig] = {
    "V0": VariantConfig(
        name="V0",
        description="Baseline: bare LLM, no tools, no RAG, no validator.",
    ),
    "V1": VariantConfig(
        name="V1",
        totaller_enabled=True,
        description="V0 + USDA-grounded foods + deterministic Totaller.",
    ),
    "V2": VariantConfig(
        name="V2",
        totaller_enabled=True,
        profile_constraints_enabled=True,
        description="V1 + profile-derived hard constraints (allergens, diet pattern).",
    ),
    "V3": VariantConfig(
        name="V3",
        totaller_enabled=True,
        profile_constraints_enabled=True,
        rag_enabled=True,
        description="V2 + clinical-guideline RAG + condition-derived rules.",
    ),
    "V4": VariantConfig(
        name="V4",
        totaller_enabled=True,
        profile_constraints_enabled=True,
        rag_enabled=True,
        reflection_enabled=True,
        description="Full system: V3 + Generate-Score-Refine reflection loop.",
    ),
}


@dataclass
class PipelineResult:
    """Bundle returned by `Pipeline.run`."""

    plan: MealPlan
    report: ValidationReport
    targets: MacroTargets
    constraints: list[HardConstraint]
    citations: list[Citation] = field(default_factory=list)
    iterations: int = 0
    variant: str = "V0"


class Pipeline:
    """End-to-end orchestrator.

    The orchestrator is *stateless* with respect to a single recommendation
    request - all per-request state lives on `AgentDeps`. This makes it safe
    to reuse one `Pipeline` across the entire ablation study run.
    """

    def __init__(
        self,
        variant: VariantConfig | str,
        *,
        model: str | None = None,
        profile_service: ProfileService | None = None,
        usda: USDAClient | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        if isinstance(variant, str):
            variant = VARIANTS[variant]
        self.variant = variant
        self._settings = get_settings()
        self._model = model or self._settings.llm_model
        self._profile_service = profile_service or ProfileService.default()
        # Lazy-init heavy collaborators; only create them when the variant needs them.
        self._usda = usda
        self._retriever = retriever

    def _ensure_usda(self) -> USDAClient:
        if self._usda is None:
            self._usda = USDAClient()
        return self._usda

    def _ensure_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def _build_constraints(
        self,
        profile: UserProfile,
        rag_citations: list[Citation],
    ) -> list[HardConstraint]:
        constraints: list[HardConstraint] = []
        if self.variant.profile_constraints_enabled:
            constraints.extend(self._profile_service.derive_hard_constraints(profile))
        # RAG-derived constraints are already baked into _CONDITION_RULES; the
        # extra rationale we add here is purely for explainability in the
        # eventual ValidationReport.
        if self.variant.rag_enabled:
            for c in constraints:
                if c.rationale is None and rag_citations:
                    object.__setattr__(c, "rationale", rag_citations[0].snippet[:160])
        return constraints

    async def _retrieve_context(self, profile: UserProfile, query: str) -> list[Citation]:
        if not self.variant.rag_enabled:
            return []
        retriever = self._ensure_retriever()
        # Build a profile-conditioned query for better recall on Level-3 cases.
        cond_terms = " ".join(c.value for c in profile.conditions)
        diet_term = profile.diet_pattern.value if profile.diet_pattern.value != "omnivore" else ""
        full_query = " ".join(filter(None, [query, cond_terms, diet_term, "dietary recommendation"]))
        return [c.to_citation() for c in retriever.retrieve(full_query)]

    async def run(
        self,
        profile: UserProfile,
        user_query: str,
        *,
        targets: MacroTargets | None = None,
    ) -> PipelineResult:
        targets = targets or derive_macro_targets(profile)
        rag_citations = await self._retrieve_context(profile, user_query)
        constraints = self._build_constraints(profile, rag_citations)

        deps = AgentDeps(
            profile=profile,
            targets=targets,
            constraints=constraints,
            profile_service=self._profile_service,
            usda=self._ensure_usda() if self.variant.needs_usda else None,
            retriever=self._ensure_retriever() if self.variant.rag_enabled else None,
        )

        agent = build_nutrition_agent(model=self._model)
        prompt = self._compose_prompt(profile, user_query, targets, constraints, rag_citations)
        result = await agent.run(prompt, deps=deps)
        plan = result.output

        # Always merge in the RAG citations so Faithfulness can be evaluated even
        # when the LLM forgot to copy them through.
        if rag_citations:
            existing = {(c.source, c.snippet) for c in plan.citations}
            for c in rag_citations:
                if (c.source, c.snippet) not in existing:
                    plan.citations.append(c)

        if self.variant.reflection_enabled:
            refl: ReflectionResult = await reflect_and_refine(plan, deps, model=self._model)
            return PipelineResult(
                plan=refl.plan,
                report=refl.report,
                targets=targets,
                constraints=constraints,
                citations=rag_citations,
                iterations=refl.iterations,
                variant=self.variant.name,
            )

        report = (
            validate_meal_plan(plan, constraints)
            if constraints
            else ValidationReport(hard_satisfied=True, totals=total_meal_plan(plan).totals)
        )
        return PipelineResult(
            plan=plan,
            report=report,
            targets=targets,
            constraints=constraints,
            citations=rag_citations,
            iterations=0,
            variant=self.variant.name,
        )

    def _compose_prompt(
        self,
        profile: UserProfile,
        user_query: str,
        targets: MacroTargets,
        constraints: list[HardConstraint],
        rag_citations: list[Citation],
    ) -> str:
        sections = [
            f"User query: {user_query}",
            f"Profile (complexity level {profile.complexity_level}):",
            profile.model_dump_json(indent=2),
            "Macro targets (single day):",
            targets.model_dump_json(indent=2),
        ]
        if constraints:
            sections.append("Hard constraints (MUST be satisfied; the validator will check):")
            sections.append(
                "\n".join(
                    f"- {c.kind}::{c.target}"
                    + (f" (<= {c.value})" if c.kind == "max_nutrient" else "")
                    + (f" (>= {c.value})" if c.kind == "min_nutrient" else "")
                    for c in constraints
                ),
            )
        else:
            sections.append("No hard constraints declared.")
        if rag_citations:
            sections.append("Clinical-guideline excerpts (use these to ground your rationale):")
            for c in rag_citations[:6]:
                sections.append(
                    f"[{c.source}{f' p.{c.page}' if c.page else ''}] {c.snippet}",
                )
        sections.append("Return ONLY a valid MealPlan object.")
        return "\n\n".join(sections)
