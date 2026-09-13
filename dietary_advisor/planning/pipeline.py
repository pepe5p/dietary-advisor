"""Top-level orchestrator with feature flags for the three ablatable modules.

The pipeline is intentionally a *deterministic Python flow* rather than an
LLM-driven Meta-Agent: that keeps module comparisons clean (no second hidden
LLM behaviour to worry about). Totaller and reflection default on; RAG is
opt-in. Each module can be toggled for ablation, on top of a baseline that
always includes the Open Food Facts food lookup and the
patient's fixed profile (allergens, conditions, goals) - see AGENTS.md for why
the food DB can't be an ablation variant (the agent cannot invent a food).

Production never checks hard nutrient/allergen constraints against incomplete
food-DB micros; the agent must infer restrictions from the profile itself, the
same way a human nutritionist would. Evaluation scores macro-target error and
soft preferences (LLM judge) only - see `evaluation.validation`.

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
That step uses `meal_idea_llm_model` from settings, not the pipeline's main
model override, so ablation sweeps do not change the brainstorm model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic_ai.models import infer_model, Model

from dietary_advisor.agents.agent_output import AgentMealPlan
from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.agents.meal_idea import build_meal_idea_agent, MealConcept
from dietary_advisor.agents.meal_idea.prompts import meal_idea_user_prompt
from dietary_advisor.agents.nutrition import build_nutrition_agent
from dietary_advisor.agents.nutrition.prompts import nutrition_user_prompt
from dietary_advisor.agents.prompt_blocks import has_user_request
from dietary_advisor.agents.rag_query import build_rag_query_agent
from dietary_advisor.agents.rag_query.prompts import rag_query_user_prompt
from dietary_advisor.agents.runner import run_agent_logged
from dietary_advisor.config import get_settings
from dietary_advisor.config.llm import LlmSpec
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

    Each flag maps 1:1 to one ablatable module.
    Totaller and reflection default on; RAG defaults off. The patient profile
    and the Open Food Facts food lookup are always active (part of the baseline
    task), so neither is a flag - see AGENTS.md.
    """

    totaller_enabled: bool = True
    rag_enabled: bool = False
    reflection_enabled: bool = True

    @property
    def label(self) -> str:
        """Short identifier for this config, used in reports and logs."""
        enabled = []
        if self.totaller_enabled:
            enabled.append("totaller")
        if self.rag_enabled:
            enabled.append("rag")
        if self.reflection_enabled:
            enabled.append("reflective-loop")
        if not enabled:
            return "baseline"
        if len(enabled) == 3:
            return "full"
        return "+".join(enabled)

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
        llm: LlmSpec | None = None,
    ) -> None:
        self.variant = variant if variant is not None else VariantConfig()
        self._settings = get_settings()
        # Lazy-init heavy collaborators; only create them when the variant needs them.
        self._food_db = food_db
        self._owns_food_db = False  # Only close DBs we created ourselves.
        self._retriever = retriever
        # Resolve once so every agent in this pipeline shares the same Model.
        spec = llm if llm is not None else self._settings.llm_spec
        self._model: Model = infer_model(spec.model)
        self._model_settings = spec.model_settings
        self._model_id = str(spec)

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

    async def _retrieve_context(
        self,
        deps: AgentDeps,
        user_query: str,
        *,
        has_request: bool,
    ) -> tuple[list[Citation], RunTelemetry]:
        if not self.variant.rag_enabled:
            log.debug("RAG disabled for variant %s; skipping retrieval.", self.variant.label)
            return [], RunTelemetry()
        retriever = self._ensure_retriever()
        queries, telemetry = await self._generate_rag_queries(deps, user_query, has_request=has_request)
        log.debug("Retrieving clinical guidelines for %d query/queries: %r", len(queries), queries)
        citations = self._retrieve_for_queries(retriever, queries)
        log.info("RAG retrieved %d citation(s) across %d query/queries.", len(citations), len(queries))
        return citations, telemetry

    async def _generate_rag_queries(
        self,
        deps: AgentDeps,
        user_query: str,
        *,
        has_request: bool,
    ) -> tuple[list[str], RunTelemetry]:
        """Ask the query agent for guideline searches, degrading to a heuristic query on failure.

        A deliberate empty result is honoured, not overridden: the query agent
        returns no queries for a plain profile with no specialized needs, and
        retrieval then stays quiet rather than pulling generic excerpts the
        baseline prompt already covers. Only an agent *failure* falls back to
        the pre-agent heuristic, so a broken brainstorm never sinks the request.
        """
        prompt = rag_query_user_prompt(deps, user_query)
        try:
            agent = build_rag_query_agent(
                model=self._model, model_settings=self._model_settings, has_user_request=has_request
            )
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
        prompt = meal_idea_user_prompt(deps, user_query, rag_citations=rag_citations)
        try:
            agent = build_meal_idea_agent()
            result = await run_agent_logged(agent, prompt, deps=deps, label="meal_idea")
        except Exception as exc:  # noqa: BLE001 - LLMs raise many things; never sink the request for this
            log.warning("Meal-idea agent failed, continuing without meal concepts: %s", exc)
            return [], RunTelemetry()
        log.info("Meal-idea agent produced %d meal concept(s).", len(result.output))
        return result.output, collect_from_result(result)

    async def run(self, profile: UserProfile, user_query: str) -> PipelineResult:
        has_request = has_user_request(user_query)
        log.info(
            "Running pipeline variant=%s query=%r",
            self.variant.label,
            user_query,
        )
        deps = AgentDeps(
            profile=profile,
            targets=profile.targets,
            food_db=self._ensure_food_db(),
            retriever=self._ensure_retriever() if self.variant.rag_enabled else None,
        )

        # Retrieval is the first step: the guideline excerpts it produces are
        # fed into every downstream agent (meal-idea, nutrition, reflection),
        # so the meal-idea brainstorm can no longer run concurrently with it.
        rag_citations, rag_telemetry = await self._retrieve_context(deps, user_query, has_request=has_request)

        meal_concepts, meal_idea_telemetry = await self._generate_meal_ideas(deps, user_query, rag_citations)

        agent = build_nutrition_agent(
            totaller_enabled=self.variant.totaller_enabled,
            rag_enabled=self.variant.rag_enabled,
            has_user_request=has_request,
            model=self._model,
            model_settings=self._model_settings,
        )
        prompt = nutrition_user_prompt(
            deps,
            user_query,
            rag_citations=rag_citations,
            meal_concepts=meal_concepts,
        )
        log.info(
            "Invoking nutrition agent (model=%s, totaller=%s)...",
            self._model_id,
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
                model=self._model,
                model_settings=self._model_settings,
                has_user_request=has_request,
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
                "cache_read_tokens=%d reasoning_tokens=%s"
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
            telemetry.reasoning_tokens or "None",
        )
        return PipelineResult(
            plan=plan,
            agent_plan=agent_plan,
            targets=profile.targets,
            citations=rag_citations,
            iterations=iterations,
            variant=self.variant.label,
            shopping_list=build_shopping_list(plan),
            telemetry=telemetry,
        )
