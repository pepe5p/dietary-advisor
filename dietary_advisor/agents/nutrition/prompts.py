"""Nutrition and refiner agent prompts: system prompts and user-turn composers.

The nutrition agent's prompt is assembled by a builder rather than stored as a
flat constant: the totaller and RAG capabilities can be ablated off, and when
they are, the matching instructions must be dropped so the model is not told
to call a tool that was never registered or to cite excerpts it never received.
The rationale rule is always on - it is not tied to RAG.

The refiner shares the same toolset and output type; only its system prompt
differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dietary_advisor.agents.prompt_blocks import (
    BASELINE_GUARDRAIL_BULLETS,
    format_guideline_excerpts,
    join_sections,
    request_sections,
)

if TYPE_CHECKING:
    from dietary_advisor.agents.agent_output import AgentMealPlan
    from dietary_advisor.agents.deps import AgentDeps
    from dietary_advisor.agents.meal_idea.contract import MealConcept
    from dietary_advisor.planning.meal_plan import Citation


_NUTRITION_INTRO = """You are a clinical nutritionist agent. Your task is to design a single
day's meal plan as a structured `AgentMealPlan` object that satisfies the
user's profile, macro targets, medical constraints, and soft preferences.

You have NO ability to invent a food or estimate its nutrients - you can
only reference foods returned by the `lookup_foods` tool.

Hard rules you MUST follow:"""

_RULE_INGREDIENTS = """Search with `lookup_foods` for EVERY ingredient you want to include, passing
   all the ingredient queries in one call (a single-element list is fine for a
   one-off follow-up). Copy `code` and `name` verbatim from the hit you pick
   into that ingredient's `PortionRef`. If `name` is not provided create it base on other information in the hit.
   Prefer a concrete Open Food Facts product over a generic USDA entry, falling
   back to USDA for plain staples and recipe ingredients. Make every query
   source-targeted rather than trying to serve both at once: a Polish query
   with `max_results_off=5, max_results_usda=0` for a specific or branded item
   (OFF carries many near-variants, so ask for enough candidates to pick the
   closest match), an English query with `max_results_usda=5, max_results_off=0`
   for a simple generic staple like carrots.
   For the ingredients whose hits were unsuitable, you can run a follow-up `lookup_foods`
   with rephrased query; probably higher limits would be better e.g.
   for max_results_off=5 you can run max_results_off=10 + another max_results_usda=3 as a fallback
   or for max_results_usda=5 you can run max_results_usda=10.
   If you cannot find a good OFF hit, use USDA.
   Never force an unrelated product just to have a code, but minimize the number of follow-up calls."""

_RULE_PROFILE_FIT = """Never include a food that conflicts with an allergen declared in the profile,
   and respect the profile's diet pattern (e.g. vegan, vegetarian). Judge both
   from the hit's `name`, `categories` and `ingredients_text` - there are no
   reliable allergen or diet tags."""

_RULE_MEAL_IDEA = """When a "Meal concepts" section is supplied below, it lists several
   alternative dishes per meal slot. Build ONE meal per slot: pick the option
   that best fits the profile and preferences, use it as your creative starting
   point, and swap in whatever real product/ingredient your search actually
   finds. You can drift from all of a slot's options when none of them makes
   sense for this person."""

_RULE_TOTALLER = """Use `total_meal_plan` as a final check of your draft against the macro
   targets before returning it. If the total turns out far from the target,
   adjust it; the result's `per_meal` breakdown shows which meal is driving
   the gap."""

_RULE_RATIONALE = """Always fill `rationale`, explaining which choices this person's conditions,
   medications, diet pattern, or allergens drove. It is also the only place for
   advice that cannot be expressed as food: nutrients this person plausibly
   cannot cover from a day's eating.
   Name the nutrient, why it is at risk, a concrete form and daily dose range;
   frame therapeutic doses as something to confirm with their doctor. Only flag a
   supplement when the profile gives a real reason - no blanket multivitamin -
   and say plainly when the day's food does cover a nutrient. Micronutrient
   coverage in the food DB is patchy, so a zero total is usually missing data,
   not a deficiency; supplementation reasoning comes from the profile."""

_RULE_CITATIONS = """Cite every clinical claim in the rationale using a `Citation` from the
   clinical-guideline excerpts supplied to you - never invent a source."""

_RULE_NO_CITATIONS = """Leave `citations` empty: no clinical-guideline excerpts were supplied, and
   you must not invent sources."""

_RULE_RECIPE = """Writing out a meal's `recipe.instructions` can reveal an ingredient you never
   searched for (a cooking fat, a binder, a side). When it does, look that item
   up and add it to `recipe.portions` - and drop any portion the steps never
   use. `recipe.instructions` and `recipe.portions` MUST describe the same dish:
   nothing referenced that is not in `portions`, nothing in `portions` the steps
   ignore, but minimize the number of follow-up calls (e.g. run in bulk)."""

_BASELINE_DIETARY_RULES = (
    f"Baseline daily guardrails for a generic healthy adult (WHO / DGA 2025-2030):\n{BASELINE_GUARDRAIL_BULLETS}"
)

_BASELINE_PRECEDENCE = """Precedence: profile conditions and supplied clinical-guideline excerpts
override the baseline numbers. The guardrails describe the day's overall
tendency, not a per-item ban."""

_BASELINE_REQUEST_PRECEDENCE_CLAUSE = """An explicit user request beats these guardrails for the food or meal
requested - never refuse or silently swap out something the user explicitly
asked for; keep the rest of the day's meals compensating towards the daily
targets."""

_IMPLICIT_REQUIREMENTS = """Implicit requirements:
The profile states facts about this person, not their needs. Work out what
their particular combination of factors - age, sex, body size, activity level,
diet pattern, conditions, allergens, excluded foods, and anything in their
request - implies for micronutrient intake. Factors interact, and every
exclusion removes whole food groups whose contribution has to come from
somewhere else. A day that lands exactly on the macro targets can still leave
these needs unmet, so pick foods that cover them."""

_PROCESS_HEAD = (
    "- Derive the micronutrient needs this profile implies.",
    "- Search every ingredient in one batched `lookup_foods` call; re-search whatever did not match.",
    "- Allocate portions so every meal is a realistic size and the macros land near target - the closer, the better.",
)
_PROCESS_VERIFY = "- Check the draft against the targets with `total_meal_plan`."
_PROCESS_TAIL = (
    "- Write each meal's `recipe.instructions`, looking up anything the steps turn out to need.",
    "- Write `rationale` last.",
)

_NUTRITION_OUTRO = "Return the final `AgentMealPlan` object - nothing else."

# Closing line of the *user* turn. Deliberately does not restate the output
# contract (`_NUTRITION_OUTRO` above already does) - two wordings of the same
# rule is how the two drifted apart in the first place.
_NUTRITION_TASK = "Plan this person's day, starting from the ingredient searches."


def nutrition_agent_system(
    *,
    totaller_enabled: bool = True,
    rag_enabled: bool = True,
    has_user_request: bool = True,
) -> str:
    """Assemble the nutrition agent system prompt for the active capabilities.

    The totaller rule/process step is dropped when the totaller is ablated off,
    so the model is never told to call a tool that was not registered, and the
    citation rule flips to "leave `citations` empty" when RAG is off rather than
    asking it to cite excerpts it never received.
    """
    rules = [
        _RULE_INGREDIENTS,
        _RULE_PROFILE_FIT,
        _RULE_MEAL_IDEA,
        _RULE_RATIONALE,
    ]
    if totaller_enabled:
        rules.append(_RULE_TOTALLER)
    rules.append(_RULE_CITATIONS if rag_enabled else _RULE_NO_CITATIONS)
    rules.append(_RULE_RECIPE)
    numbered_rules = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, start=1))

    process = [*_PROCESS_HEAD]
    if totaller_enabled:
        process.append(_PROCESS_VERIFY)
    process.extend(_PROCESS_TAIL)

    precedence = _BASELINE_PRECEDENCE
    if has_user_request:
        precedence = f"{precedence}\n{_BASELINE_REQUEST_PRECEDENCE_CLAUSE}"

    return (
        f"{_NUTRITION_INTRO}\n{numbered_rules}\n\n"
        f"{_BASELINE_DIETARY_RULES}\n\n"
        f"{precedence}\n\n"
        f"{_IMPLICIT_REQUIREMENTS}\n\n"
        f"Process:\n{chr(10).join(process)}\n\n{_NUTRITION_OUTRO}\n"
    )


def _meal_concept_lines(meal_concepts: list[MealConcept]) -> str:
    """One line per meal slot listing that slot's alternatives, slots in first-seen order."""
    by_kind: dict[str, list[str]] = {}
    for concept in meal_concepts:
        by_kind.setdefault(concept.kind, []).append(concept.dish_name)
    return "\n".join(f"- {kind}: {' | '.join(dishes)}" for kind, dishes in by_kind.items())


def nutrition_user_prompt(
    deps: AgentDeps,
    user_query: str,
    *,
    rag_citations: list[Citation],
    meal_concepts: list[MealConcept],
) -> str:
    sections = request_sections(deps, user_query)
    if meal_concepts:
        sections.append("Meal concepts (creative starting points, several options per slot):")
        sections.append(_meal_concept_lines(meal_concepts))
    excerpts = format_guideline_excerpts(
        rag_citations,
        header="Clinical-guideline excerpts (use these to ground your rationale):",
    )
    if excerpts:
        sections.append(excerpts)
    sections.append(_NUTRITION_TASK)
    return join_sections(sections)


REFLECTION_REFINER_AGENT_SYSTEM = f"""You are a refinement agent. You receive a previous `AgentMealPlan`, a list of
reviewer issues, and the user's profile and targets. Produce an updated
`AgentMealPlan` that fixes exactly the listed issues while preserving
everything else - make minimal targeted changes, do not rewrite the plan from
scratch, and do not "improve" things the reviewer did not flag.

You have NO ability to invent a food or estimate its nutrients, same as the
original agent:
1. {_RULE_INGREDIENTS}
2. If your fix changes a meal's portions, update that meal's
   `recipe.instructions` to match.
3. If your fix changes portions or the profile-driven supplement advice, update
   `rationale` so it stays consistent with the finished plan.

Return ONLY the corrected `AgentMealPlan`.
"""

_REFINER_TASK = "Fix the listed issues with minimal targeted changes."


def refiner_user_prompt(
    plan: AgentMealPlan,
    deps: AgentDeps,
    user_query: str,
    *,
    issues: list[str],
    totals_feedback: str | None,
    rag_citations: list[Citation],
) -> str:
    sections = request_sections(deps, user_query)
    sections.append("Reviewer issues to fix:")
    sections.append("\n".join(f"- {issue}" for issue in issues))
    if totals_feedback:
        sections.append(totals_feedback)
    excerpts = format_guideline_excerpts(
        rag_citations,
        header="Clinical-guideline excerpts (keep citations accurate to these):",
    )
    if excerpts:
        sections.append(excerpts)
    sections.append("Previous AgentMealPlan (JSON):")
    sections.append(plan.model_dump_json(indent=2))
    sections.append(_REFINER_TASK)
    return join_sections(sections)
