"""Shared prompt fragments used by every agent's user-turn builder.

Each agent package owns its system prompt and user-turn composer; the pieces
that every composer needs - request/profile/targets head, guideline excerpts,
totals-vs-targets feedback - live here so they stay identical across agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dietary_advisor.totaller.nutrition import NutrientName

if TYPE_CHECKING:
    from dietary_advisor.agents.deps import AgentDeps
    from dietary_advisor.planning.meal_plan import Citation, NutrientTotals
    from dietary_advisor.totaller.nutrition import MacroTargets


# Baseline numeric guardrails distilled from WHO / DGA 2025-2030. Stated in the
# prompt (rather than retrieved) because they apply to every healthy adult, so
# spending RAG excerpts on them would starve the condition-specific retrieval.
# Kept out on purpose (they arrive via RAG when the profile warrants): DASH's
# 1500 mg sodium target, diabetes 15 g carbohydrate exchanges, the DGA
# dairy-snack sugar rule.
# Shared by the planner (which must respect them) and the critic (which checks
# them), so the two can never drift to different numbers.
BASELINE_GUARDRAIL_BULLETS = """- Fat: < 30% of energy; saturated fat < 10% of energy; trans fat < 1% of energy.
- Free sugars: < 10% of energy (~50 g at 2000 kcal); added sugars max 10 g per meal.
- Sodium: < 2000 mg/day (~5 g salt); never exceed 2300 mg/day.
- Potassium: >= 3.5 g/day.
- Fruit + vegetables: >= 400 g/day (~5 portions; potatoes/starchy roots do not count)."""


def format_guideline_excerpts(citations: list[Citation], *, header: str) -> str | None:
    """Render RAG citations as one prompt block, or `None` when there are none.

    Every agent package routes its excerpts through this, so the same chunks
    reach the nutrition, meal-idea, critic and refiner agents in one format.
    """
    if not citations:
        return None
    lines = [header]
    for c in citations:
        page = f" p.{c.page}" if c.page else ""
        lines.append(f"[{c.source}{page}] {c.snippet}")
    return "\n".join(lines)


def _fmt_pct(actual: float, target: float) -> str:
    if target <= 0:
        return "n/a"
    return f"{(actual - target) / target * 100:+.0f}%"


def format_totals_feedback(totals: NutrientTotals, targets: MacroTargets) -> str:
    """Render the deterministic totals-vs-targets block injected into critic/refiner prompts."""
    t = totals.totals
    target_map = targets.as_dict()

    def part(name: NutrientName, label: str, unit: str) -> str:
        actual = t.get(name, 0.0)
        target = target_map[name]
        return f"{label} {actual:.0f} {unit} (target {target:.0f}, {_fmt_pct(actual, target)})"

    overall = ", ".join(
        [
            part(NutrientName.ENERGY_KCAL, "", "kcal").strip(),
            part(NutrientName.PROTEIN_G, "protein", "g"),
            part(NutrientName.CARBS_G, "carbs", "g"),
            part(NutrientName.FAT_G, "fat", "g"),
            part(NutrientName.FIBER_G, "fiber", "g"),
        ]
    )
    per_meal_lines = [
        (
            f'- {meal.kind} "{meal.name}": {meal.get(NutrientName.ENERGY_KCAL):.0f} kcal, '
            f"protein {meal.get(NutrientName.PROTEIN_G):.0f} g, "
            f"carbs {meal.get(NutrientName.CARBS_G):.0f} g, "
            f"fat {meal.get(NutrientName.FAT_G):.0f} g"
        )
        for meal in totals.per_meal
    ]
    lines = [
        "Deterministic nutrient totals (computed by the system, trust these numbers):",
        f"Overall: {overall}",
        "Per meal:",
        *per_meal_lines,
    ]
    if totals.warnings:
        lines.extend(
            [
                "Data-coverage caveats (totals above are understated for these nutrients):",
                *(f"- {warning}" for warning in totals.warnings),
            ],
        )
    return "\n".join(lines)


def has_user_request(user_query: str) -> bool:
    return bool(user_query.strip())


def request_sections(deps: AgentDeps, user_query: str) -> list[str]:
    """The request/profile/targets head every agent's user turn starts from."""
    sections = [f"User request: {user_query}"] if has_user_request(user_query) else []
    sections.extend(
        [
            "Profile:",
            deps.profile.model_dump_json(indent=2),
            "Macro targets (single day):",
            deps.targets.model_dump_json(indent=2),
        ],
    )
    return sections


def join_sections(sections: list[str]) -> str:
    return "\n\n".join(sections)
