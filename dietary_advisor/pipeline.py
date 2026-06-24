"""Top-level orchestrator with feature flags for the four supporting modules.

The pipeline is intentionally a *deterministic Python flow* rather than an
LLM-driven Meta-Agent: that keeps module comparisons clean (no second hidden
LLM behaviour to worry about). By default all four modules are enabled (the
full system); each can be individually disabled for ablation, on top of a
baseline LLM that always considers the patient's fixed profile (allergens,
conditions, goals).

Production never derives or checks hard constraints - that is an
evaluation-only concept (see `evaluation.constraints`/`evaluation.validation`);
the agent must infer restrictions from the profile itself, the same way a
human nutritionist would.

The four toggleable modules (desc.md):
    Food DB    - Open Food Facts food data (truth source for nutrients)
    Totaller   - deterministic nutrient summation, exposed to the agent as a tool
    RAG        - clinical-guideline retrieval
    Reflection - Generate-Review-Refine self-correction loop (plain self-review)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.nutrition_agent import build_nutrition_agent
from dietary_advisor.agents.runner import run_agent_logged
from dietary_advisor.config import get_settings
from dietary_advisor.knowledge.retriever import HybridRetriever
from dietary_advisor.llm import resolve_llm_model
from dietary_advisor.schemas.meal_plan import Citation, MealPlan, ShoppingList
from dietary_advisor.schemas.nutrition import MacroTargets
from dietary_advisor.schemas.profile import UserProfile
from dietary_advisor.telemetry import collect_from_result, RunTelemetry
from dietary_advisor.tools.food_db import OffFoodDb
from dietary_advisor.tools.shopping_list import build_shopping_list
from dietary_advisor.validation.reflection import reflect_and_refine, ReflectionResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantConfig:
    """Feature flags controlling which supporting modules are active.

    Each flag maps 1:1 to one of the four supporting modules in `desc.md`. All
    default to enabled (the full system). The patient profile is always
    considered (it is part of the baseline task), so it is not a flag.
    """

    food_enabled: bool = True
    totaller_enabled: bool = True
    rag_enabled: bool = True
    reflection_enabled: bool = True

    @property
    def needs_food_db(self) -> bool:
        return self.food_enabled

    @property
    def label(self) -> str:
        """Short identifier for this config, used in reports and logs."""
        disabled = []
        if not self.food_enabled:
            disabled.append("no-off")
        if not self.totaller_enabled:
            disabled.append("no-totaller")
        if not self.rag_enabled:
            disabled.append("no-rag")
        if not self.reflection_enabled:
            disabled.append("no-reflective-loop")
        if not disabled:
            return "full"
        if len(disabled) == 4:
            return "baseline"
        return "+".join(disabled)

    @property
    def description(self) -> str:
        """Human-readable list of the modules enabled in this config."""
        enabled = []
        if self.food_enabled:
            enabled.append("Food DB")
        if self.totaller_enabled:
            enabled.append("Totaller")
        if self.rag_enabled:
            enabled.append("RAG")
        if self.reflection_enabled:
            enabled.append("Reflection loop")
        return f"Enabled: {', '.join(enabled)}." if enabled else "Baseline: no symbolic modules enabled."


@dataclass
class PipelineResult:
    """Bundle returned by `Pipeline.run`."""

    plan: MealPlan
    targets: MacroTargets
    citations: list[Citation] = field(default_factory=list)
    iterations: int = 0
    variant: str = "full"
    shopping_list: ShoppingList = field(default_factory=ShoppingList)
    # Full nutrition-agent conversation, threaded back in as `message_history`
    # on the next turn to support multi-turn discussion / plan revision.
    messages: list[ModelMessage] = field(default_factory=list)
    # Tool-call counts and token usage across the main agent run + reflection
    # loop, surfaced by the CLI under `--verbose`.
    telemetry: RunTelemetry = field(default_factory=RunTelemetry)


class Pipeline:
    """End-to-end orchestrator.

    The orchestrator is *stateless* with respect to a single recommendation
    request - all per-request state lives on `AgentDeps`. This makes it safe
    to reuse one `Pipeline` across the entire ablation study run.
    """

    def __init__(
        self,
        variant: VariantConfig | None = None,
        *,
        model: str | None = None,
        food_db: OffFoodDb | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.variant = variant if variant is not None else VariantConfig()
        self._settings = get_settings()
        self._model = resolve_llm_model(model or self._settings.llm_model)
        # Lazy-init heavy collaborators; only create them when the variant needs them.
        self._food_db = food_db
        self._owns_food_db = False  # Only close the food DB we created ourselves.
        self._retriever = retriever

    def _ensure_food_db(self) -> OffFoodDb:
        if self._food_db is None:
            self._food_db = OffFoodDb()
            self._owns_food_db = True
        return self._food_db

    def _ensure_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def close(self) -> None:
        """Release any lazily-created collaborators (currently the food DB).

        Only collaborators the pipeline created itself are closed; ones passed
        in through the constructor are left untouched, since their lifecycle
        is owned by the caller. Idempotent.
        """
        if self._owns_food_db and self._food_db is not None:
            self._food_db.close()
            self._food_db = None
            self._owns_food_db = False

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def _retrieve_context(self, profile: UserProfile, query: str) -> list[Citation]:
        if not self.variant.rag_enabled:
            log.debug("RAG disabled for variant %s; skipping retrieval.", self.variant.label)
            return []
        retriever = self._ensure_retriever()
        # Build a profile-conditioned query for better recall on Level-3 cases.
        cond_terms = " ".join(profile.conditions)
        diet_term = profile.diet_pattern if profile.diet_pattern != "omnivore" else ""
        full_query = " ".join(filter(None, [query, cond_terms, diet_term, "dietary recommendation"]))
        log.debug("Retrieving clinical guidelines for query: %r", full_query)
        citations = [c.to_citation() for c in retriever.retrieve(full_query)]
        log.info("RAG retrieved %d citation(s).", len(citations))
        return citations

    async def run(
        self,
        profile: UserProfile,
        user_query: str,
        *,
        targets: MacroTargets | None = None,
        available_ingredients: list[str] | None = None,
        message_history: list[ModelMessage] | None = None,
    ) -> PipelineResult:
        targets = targets or profile.targets
        is_followup = bool(message_history)
        log.info(
            "Running pipeline variant=%s follow_up=%s query=%r",
            self.variant.label,
            is_followup,
            user_query,
        )
        rag_citations = await self._retrieve_context(profile, user_query)

        deps = AgentDeps(
            profile=profile,
            targets=targets,
            food_db=self._ensure_food_db() if self.variant.needs_food_db else None,
            retriever=self._ensure_retriever() if self.variant.rag_enabled else None,
        )

        agent = build_nutrition_agent(model=self._model, totaller_enabled=self.variant.totaller_enabled)
        # A follow-up turn (message_history present) already carries the profile,
        # targets and previous plan in context, so we send a lean revision prompt
        # instead of re-stating everything.
        if is_followup:
            prompt = self._compose_followup_prompt(user_query, available_ingredients=available_ingredients)
        else:
            prompt = self._compose_prompt(
                profile,
                user_query,
                targets,
                rag_citations,
                available_ingredients=available_ingredients,
            )
        log.info("Invoking nutrition agent (model=%s, totaller=%s)...", self._model, self.variant.totaller_enabled)
        result = await run_agent_logged(agent, prompt, deps=deps, label="nutrition", message_history=message_history)
        plan = result.output
        messages = list(result.all_messages())
        telemetry = collect_from_result(result)
        log.info("Agent returned plan with %d meal(s), %d citation(s).", len(plan.meals), len(plan.citations))

        # Always merge in the RAG citations so grounding can be evaluated even
        # when the LLM forgot to copy them through.
        if rag_citations:
            existing = {(c.source, c.snippet) for c in plan.citations}
            for c in rag_citations:
                if (c.source, c.snippet) not in existing:
                    plan.citations.append(c)

        iterations = 0
        if self.variant.reflection_enabled:
            log.info("Starting reflection loop...")
            refl: ReflectionResult = await reflect_and_refine(
                plan,
                deps,
                model=self._model,
                totaller_enabled=self.variant.totaller_enabled,
            )
            plan = refl.plan
            iterations = refl.iterations
            telemetry = telemetry.merge(refl.telemetry)
            log.info("Reflection loop finished after %d iteration(s).", iterations)

        log.info(
            "Pipeline complete: variant=%s meals=%d citations=%d iterations=%d requests=%d tool_calls=%d",
            self.variant.label,
            len(plan.meals),
            len(plan.citations),
            iterations,
            telemetry.requests,
            telemetry.total_tool_calls,
        )
        return PipelineResult(
            plan=plan,
            targets=targets,
            citations=rag_citations,
            iterations=iterations,
            variant=self.variant.label,
            shopping_list=build_shopping_list(plan),
            messages=messages,
            telemetry=telemetry,
        )

    def _compose_prompt(
        self,
        profile: UserProfile,
        user_query: str,
        targets: MacroTargets,
        rag_citations: list[Citation],
        *,
        available_ingredients: list[str] | None = None,
    ) -> str:
        sections = [
            f"User query: {user_query}",
            "Profile:",
            profile.model_dump_json(indent=2),
            "Macro targets (single day):",
            targets.model_dump_json(indent=2),
        ]
        if available_ingredients:
            sections.append("Available ingredients to use first (the user has these on hand):")
            sections.append("\n".join(f"- {name}" for name in available_ingredients))
        if rag_citations:
            sections.append("Clinical-guideline excerpts (use these to ground your rationale):")
            for c in rag_citations[:6]:
                sections.append(
                    f"[{c.source}{f' p.{c.page}' if c.page else ''}] {c.snippet}",
                )
        sections.append("Return ONLY a valid MealPlan object.")
        return "\n\n".join(sections)

    def _compose_followup_prompt(
        self,
        user_query: str,
        *,
        available_ingredients: list[str] | None = None,
    ) -> str:
        sections = [
            f"Follow-up request: {user_query}",
            "Revise the current meal plan to satisfy this request while keeping the "
            "rest of the plan intact. Make minimal targeted changes.",
        ]
        if available_ingredients:
            sections.append("Available ingredients to use first:")
            sections.append("\n".join(f"- {name}" for name in available_ingredients))
        sections.append("Return ONLY the full, updated MealPlan object.")
        return "\n\n".join(sections)
