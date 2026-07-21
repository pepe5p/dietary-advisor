"""Top-level orchestrator with feature flags for the three ablatable modules.

The pipeline is intentionally a *deterministic Python flow* rather than an
LLM-driven Meta-Agent: that keeps module comparisons clean (no second hidden
LLM behaviour to worry about). By default all three modules are enabled (the
full system); each can be individually disabled for ablation, on top of a
baseline that always includes the Open Food Facts food lookup and the
patient's fixed profile (allergens, conditions, goals) - see AGENTS.md for why
the food DB can't be an ablation variant (the agent cannot invent a food).

Production never derives or checks hard constraints - that is an
evaluation-only concept (see `evaluation.constraints`/`evaluation.validation`);
the agent must infer restrictions from the profile itself, the same way a
human nutritionist would.

The three toggleable modules (desc.md):
    Totaller   - deterministic nutrient summation, exposed to the agent as a
                 tool during generation and, when reflection is also on, as
                 the grounding for the critic's feedback (see
                 `dietary_advisor.reflection`)
    RAG        - clinical-guideline retrieval. A tool-less query agent turns
                 the request and profile into targeted searches (scaled to
                 profile complexity), the hybrid retriever runs each, and the
                 fused, deduped excerpts are fed to every downstream agent
                 (meal-idea, nutrition, critic, refiner). This runs first so
                 all later steps see the same grounding.
    Reflection - critique-then-refine self-correction loop: a tool-less critic
                 agent reviews the plan and the refiner only runs when it
                 reports issues

A further, always-on step (not a toggleable module - see AGENTS.md/the
meal-idea agent) runs a lightweight brainstorming agent after retrieval and
ahead of the nutrition agent, so the nutrition agent's ingredient choices
start from a concrete, varied dish concept instead of an abstract macro gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.meal_idea import MealConcept
from dietary_advisor.agents.meal_idea_agent import build_meal_idea_agent
from dietary_advisor.agents.nutrition_agent import build_nutrition_agent
from dietary_advisor.agents.prompts import format_guideline_excerpts
from dietary_advisor.agents.rag_query_agent import build_rag_query_agent
from dietary_advisor.agents.runner import run_agent_logged
from dietary_advisor.config import get_settings
from dietary_advisor.dietary_rag.retriever import HybridRetriever, RetrievedChunk
from dietary_advisor.food_db import FoodDb
from dietary_advisor.planning.hydration import hydrate_meal_plan
from dietary_advisor.planning.meal_plan import Citation, MealPlan, ShoppingList
from dietary_advisor.planning.shopping_list import build_shopping_list
from dietary_advisor.profile import UserProfile
from dietary_advisor.reflection import reflect_and_refine, ReflectionResult
from dietary_advisor.telemetry import collect_from_result, RunTelemetry
from dietary_advisor.totaller.nutrition import MacroTargets

log = logging.getLogger(__name__)

# Upper bound on guideline excerpts fed to the agents, regardless of how many
# queries the RAG query agent produced - keeps the prompt from ballooning on a
# complex, many-condition profile while still letting the excerpt budget grow
# past `rag_top_k` when several queries are in play.
_MAX_EXCERPTS = 12


@dataclass(frozen=True)
class VariantConfig:
    """Feature flags controlling which supporting modules are active.

    Each flag maps 1:1 to one of the three ablatable modules in `desc.md`. All
    default to enabled (the full system). The patient profile and the Open
    Food Facts food lookup are always active (part of the baseline task), so
    neither is a flag - see AGENTS.md.
    """

    totaller_enabled: bool = True
    rag_enabled: bool = True
    reflection_enabled: bool = True

    @property
    def label(self) -> str:
        """Short identifier for this config, used in reports and logs."""
        disabled = []
        if not self.totaller_enabled:
            disabled.append("no-totaller")
        if not self.rag_enabled:
            disabled.append("no-rag")
        if not self.reflection_enabled:
            disabled.append("no-reflective-loop")
        if not disabled:
            return "full"
        if len(disabled) == 3:
            return "baseline"
        return "+".join(disabled)

    @property
    def description(self) -> str:
        """Human-readable list of the modules enabled in this config."""
        enabled = []
        if self.totaller_enabled:
            enabled.append("Totaller")
        if self.rag_enabled:
            enabled.append("RAG")
        if self.reflection_enabled:
            enabled.append("Reflection loop")
        if not enabled:
            return "Baseline: food lookup only, no ablation modules enabled."
        return f"Enabled: {', '.join(enabled)}."


@dataclass
class PipelineResult:
    """Bundle returned by `Pipeline.run`."""

    plan: MealPlan
    # The agent's raw reference-only output (post reflection), before
    # hydration; this is what the evaluation harness scores.
    agent_plan: AgentMealPlan
    targets: MacroTargets
    citations: list[Citation] = field(default_factory=list)
    iterations: int = 0
    variant: str = "full"
    shopping_list: ShoppingList = field(default_factory=ShoppingList)
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
        food_db: FoodDb | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.variant = variant if variant is not None else VariantConfig()
        self._settings = get_settings()
        # Lazy-init heavy collaborators; only create them when the variant needs them.
        self._food_db = food_db
        self._owns_food_db = False  # Only close DBs we created ourselves.
        self._retriever = retriever

    def _ensure_food_db(self) -> FoodDb:
        """Open the food DB facade (both sources, per their usage settings)."""
        if self._food_db is None:
            self._food_db = FoodDb.open(self._settings)
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

    async def _retrieve_context(self, deps: AgentDeps, user_query: str) -> tuple[list[Citation], RunTelemetry]:
        if not self.variant.rag_enabled:
            log.debug("RAG disabled for variant %s; skipping retrieval.", self.variant.label)
            return [], RunTelemetry()
        retriever = self._ensure_retriever()
        queries, telemetry = await self._generate_rag_queries(deps, user_query)
        log.debug("Retrieving clinical guidelines for %d query/queries: %r", len(queries), queries)
        citations = self._retrieve_for_queries(retriever, queries)
        log.info("RAG retrieved %d citation(s) across %d query/queries.", len(citations), len(queries))
        return citations, telemetry

    async def _generate_rag_queries(self, deps: AgentDeps, user_query: str) -> tuple[list[str], RunTelemetry]:
        """Ask the query agent for guideline searches, degrading to a heuristic query on failure.

        A deliberate empty result is honoured, not overridden: the query agent
        returns no queries for a plain profile with no specialized needs, and
        retrieval then stays quiet rather than pulling generic excerpts the
        baseline prompt already covers. Only an agent *failure* falls back to
        the pre-agent heuristic, so a broken brainstorm never sinks the request.
        """
        prompt = self._compose_rag_query_prompt(deps.profile, user_query)
        try:
            agent = build_rag_query_agent()
            result = await run_agent_logged(agent, prompt, deps=deps, label="rag_query")
        except Exception as exc:  # noqa: BLE001 - LLMs raise many things; never sink the request for this
            log.warning("RAG query agent failed, falling back to heuristic query: %s", exc)
            return [self._heuristic_query(deps.profile, user_query)], RunTelemetry()
        queries = [q.strip() for q in result.output.queries if q.strip()]
        log.info("RAG query agent produced %d query/queries.", len(queries))
        return queries, collect_from_result(result)

    def _retrieve_for_queries(self, retriever: HybridRetriever, queries: list[str]) -> list[Citation]:
        """Run each query, dedupe chunks by id keeping the best fused score, then rank and cap.

        The cap grows with the query count (roughly two excerpts per query) so
        a complex, many-condition profile surfaces more guidance than a simple
        one, bounded by `_MAX_EXCERPTS`.
        """
        best: dict[str, RetrievedChunk] = {}
        for query in queries:
            for chunk in retriever.retrieve(query):
                existing = best.get(chunk.id)
                if existing is None or chunk.score > existing.score:
                    best[chunk.id] = chunk
        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        cap = min(_MAX_EXCERPTS, max(self._settings.rag_top_k, 2 * len(queries)))
        return [chunk.to_citation() for chunk in ranked[:cap]]

    @staticmethod
    def _heuristic_query(profile: UserProfile, user_query: str) -> str:
        cond_terms = " ".join(profile.conditions)
        diet_term = profile.diet_pattern if profile.diet_pattern != "omnivore" else ""
        return " ".join(filter(None, [user_query, cond_terms, diet_term, "dietary recommendation"]))

    async def _generate_meal_ideas(
        self,
        deps: AgentDeps,
        user_query: str,
        rag_citations: list[Citation],
    ) -> tuple[list[MealConcept], RunTelemetry]:
        """Brainstorm dish concepts for the day, degrading to an empty list on failure.

        A creative aid, not a correctness-critical step: if the call fails
        (rate limit, malformed output after exhausting retries), the nutrition
        agent simply falls back to choosing its own ingredients, so a broken
        brainstorm never sinks the whole request.
        """
        prompt = self._compose_meal_idea_prompt(deps.profile, deps.targets, user_query, rag_citations)
        try:
            agent = build_meal_idea_agent()
            result = await run_agent_logged(agent, prompt, deps=deps, label="meal_idea")
        except Exception as exc:  # noqa: BLE001 - LLMs raise many things; never sink the request for this
            log.warning("Meal-idea agent failed, continuing without meal concepts: %s", exc)
            return [], RunTelemetry()
        log.info("Meal-idea agent produced %d meal concept(s).", len(result.output))
        return result.output, collect_from_result(result)

    async def run(
        self,
        profile: UserProfile,
        user_query: str,
        *,
        targets: MacroTargets | None = None,
        available_ingredients: list[str] | None = None,
    ) -> PipelineResult:
        targets = targets or profile.targets
        log.info(
            "Running pipeline variant=%s query=%r",
            self.variant.label,
            user_query,
        )
        deps = AgentDeps(
            profile=profile,
            targets=targets,
            food_db=self._ensure_food_db(),
            retriever=self._ensure_retriever() if self.variant.rag_enabled else None,
        )

        # Retrieval is the first step: the guideline excerpts it produces are
        # fed into every downstream agent (meal-idea, nutrition, reflection),
        # so the meal-idea brainstorm can no longer run concurrently with it.
        rag_citations, rag_telemetry = await self._retrieve_context(deps, user_query)

        meal_concepts, meal_idea_telemetry = await self._generate_meal_ideas(deps, user_query, rag_citations)

        agent = build_nutrition_agent(
            totaller_enabled=self.variant.totaller_enabled,
            rag_enabled=self.variant.rag_enabled,
        )
        prompt = self._compose_prompt(
            profile,
            user_query,
            targets,
            rag_citations,
            meal_concepts,
            available_ingredients=available_ingredients,
        )
        log.info(
            "Invoking nutrition agent (model=%s, totaller=%s)...",
            self._settings.resolved_llm_model,
            self.variant.totaller_enabled,
        )
        result = await run_agent_logged(agent, prompt, deps=deps, label="nutrition")
        agent_plan: AgentMealPlan = result.output
        telemetry = collect_from_result(result).merge(meal_idea_telemetry).merge(rag_telemetry)
        log.info(
            "Agent returned plan with %d meal(s), %d citation(s).",
            len(agent_plan.meals),
            len(agent_plan.citations),
        )

        # Always merge in the RAG citations so grounding can be evaluated even
        # when the LLM forgot to copy them through.
        if rag_citations:
            existing = {(c.source, c.snippet) for c in agent_plan.citations}
            for c in rag_citations:
                if (c.source, c.snippet) not in existing:
                    agent_plan.citations.append(c)

        iterations = 0
        if self.variant.reflection_enabled:
            log.info("Starting reflection loop...")
            refl: ReflectionResult = await reflect_and_refine(
                agent_plan,
                deps,
                user_query,
                totaller_enabled=self.variant.totaller_enabled,
                rag_citations=rag_citations,
            )
            agent_plan = refl.plan
            iterations = refl.iterations
            telemetry = telemetry.merge(refl.telemetry)
            log.info(
                "Reflection loop finished after %d iteration(s), approved=%s.",
                iterations,
                refl.approved,
            )

        # Hydration is the one place the agent's references become real,
        # DB-verified FoodItems - everything downstream (shopping list, CLI
        # display, totals) operates on the hydrated MealPlan.
        plan = hydrate_meal_plan(agent_plan, deps.food_db)

        log.info(
            (
                "Pipeline complete: variant=%s meals=%d citations=%d iterations=%d requests=%d "
                "tool_calls=%d input_tokens=%d output_tokens=%d total_tokens=%d "
                "cache_read_tokens=%d reasoning_tokens=%d"
            ),
            self.variant.label,
            len(plan.meals),
            len(plan.citations),
            iterations,
            telemetry.requests,
            telemetry.total_tool_calls,
            telemetry.input_tokens,
            telemetry.output_tokens,
            telemetry.total_tokens,
            telemetry.cache_read_tokens,
            telemetry.reasoning_tokens,
        )
        return PipelineResult(
            plan=plan,
            agent_plan=agent_plan,
            targets=targets,
            citations=rag_citations,
            iterations=iterations,
            variant=self.variant.label,
            shopping_list=build_shopping_list(plan),
            telemetry=telemetry,
        )

    def _compose_prompt(
        self,
        profile: UserProfile,
        user_query: str,
        targets: MacroTargets,
        rag_citations: list[Citation],
        meal_concepts: list[MealConcept],
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
        if meal_concepts:
            sections.append("Meal concepts (creative starting points):")
            sections.append("\n".join(f"- {c.kind}: {c.dish_name}" for c in meal_concepts))
        if available_ingredients:
            sections.append("Available ingredients to use first (the user has these on hand):")
            sections.append("\n".join(f"- {name}" for name in available_ingredients))
        excerpts = format_guideline_excerpts(
            rag_citations,
            header="Clinical-guideline excerpts (use these to ground your rationale):",
        )
        if excerpts:
            sections.append(excerpts)
        sections.append("Return ONLY a valid AgentMealPlan object.")
        return "\n\n".join(sections)

    def _compose_rag_query_prompt(self, profile: UserProfile, user_query: str) -> str:
        return "\n\n".join(
            [
                f"User query: {user_query}",
                "Profile:",
                profile.model_dump_json(indent=2),
                "Produce the clinical-guideline search queries for this request.",
            ],
        )

    def _compose_meal_idea_prompt(
        self,
        profile: UserProfile,
        targets: MacroTargets,
        user_query: str,
        rag_citations: list[Citation],
    ) -> str:
        sections = [
            f"User query: {user_query}",
            "Profile:",
            profile.model_dump_json(indent=2),
            "Macro targets (single day):",
            targets.model_dump_json(indent=2),
        ]
        excerpts = format_guideline_excerpts(
            rag_citations,
            header="Clinical-guideline excerpts (keep the dish concepts consistent with these):",
        )
        if excerpts:
            sections.append(excerpts)
        sections.append("Propose three dishes concepts per meal slot.")
        return "\n\n".join(sections)
